# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import math
import os
from pathlib import Path
from itertools import chain
from typing import Optional, Literal, Union

import numpy as np
import torch
from federatedhealth.nlp_models import XLMRobertaModel
from federatedhealth.config import load_config

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)

from datasets import load_dataset
from accelerate import Accelerator, DistributedType
#from accelerate.logging import get_logger
#from accelerate.utils import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NLPLearner(Learner):
    def __init__(
        self,
        #data_path: str,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.train_task_name = train_task_name
        # client ID
        self.client_id = None
        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0
        self.model = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # when a run starts, this is where the actual settings get initialized for trainer
        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )
        
        self.model = XLMRobertaModel()
        self.config = load_config()
        data_config = self.config.data_config
        
        self.training_data_path = data_config.training_data
        self.dev_data_path = data_config.dev_data
        self.test_data_path = data_config.test_data
        
        #train_dataset_path = os.path.join(self.data_path, self.client_id + "_train.txt")
        #dev_dataset_path = os.path.join(self.data_path, self.client_id + "_dev.txt")
        #test_dataset_path = os.path.join(self.data_path, self.client_id + "_test.txt")
        self.model.initialize(app_dir, self.training_data_path, self.dev_data_path, self.test_data_path)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Typical training task pipeline
        Get global model weights (potentially with HE)
        Local training
        Return updated weights (model_diff)
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.model.train_dataloader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        processed_epochs = current_round*self.model.aggregation_epochs
        
        for epoch in range(self.model.aggregation_epochs):
            steps_to_this_epoch = processed_epochs + epoch*epoch_len
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
                
            self.model.train()
            
            self.epoch_global = self.epoch_of_start_time + epoch
            
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.model.aggregation_epochs}",
            )
            for i, batch_data in enumerate(self.model.train_dataloader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                current_step = steps_to_this_epoch + i
                self.model.fit_batch(batch_data, current_step)
        
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.model.aggregation_epochs

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Typical validation task pipeline
        Get global model weights (potentially with HE)
        Validation on local data
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # validation on global model
        model_owner = "global_model"

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        # before_train_validate only, can extend to other validate types
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        processed_epochs = current_round*self.model.aggregation_epochs
        current_step = len(self.model.train_dataloader)*processed_epochs
        
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_metric = self.model.local_valid(current_step)
            
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_perplexity_global_model ({model_owner}): {global_metric:.4f}")
            # validation metrics will be averaged with weights at server end for best model record
            metric_dxo = DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_metric},
                meta={},
            )
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.model.eval_dataloader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
