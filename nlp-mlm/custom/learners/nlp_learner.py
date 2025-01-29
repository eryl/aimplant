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
from custom.models.nlp_models import XLMRobertaModel
from custom.configs.training import TrainingArgs
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
        data_path: str,
        model_path: str,
        config_path: str,
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        num_labels: int = 3,
        ignore_token: int = -100,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Supervised NLP task Learner.
            This provides the basic functionality of a local learner for NLP models: perform before-train
            validation on global model at the beginning of each round, perform local training,
            and send the updated weights. No model will be saved locally, tensorboard record for
            local loss and global model validation score.

        Args:
            data_path: path to dataset,
            learning_rate,
            batch_size,
            model_name: the model name to be used in the pipeline
            num_labels: num_labels for the model,
            ignore_token: the value for representing padding / null token
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        super().__init__()
        self.aggregation_epochs = aggregation_epochs
        self.train_task_name = train_task_name
        self.model_path = model_path
        self.num_labels = num_labels
        self.ignore_token = ignore_token
        self.lr = learning_rate
        self.bs = batch_size
        self.data_path = data_path
        # client ID
        self.client_id = None
        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0
        # Training-related
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.device = None
        self.model = None
        self.writer = None
        self.best_metric = 0.0
        self.labels_to_ids = None
        self.ids_to_labels = None
        self.args = TrainingArgs()

    def load_data(self):
        train_dataset_path = os.path.join(self.data_path, self.client_id + "_train.txt")
        #train_dataset = load_dataset(train_dataset_path)
        dev_dataset_path = os.path.join(self.data_path, self.client_id + "_dev.txt")
        test_dataset_path = os.path.join(self.data_path, self.client_id + "_test.txt")
        #dev_dataset = load_dataset(dev_dataset_path)
        data_files = {'train': train_dataset_path, 'dev': dev_dataset_path, 'test': test_dataset_path}
        extension = 'text'
        dataset = load_dataset(extension, data_files=data_files)
        return dataset

    def get_labels(self, df_train):
        labels = []
        for x in df_train["labels"].values:
            labels.extend(x.split(" "))
        unique_labels = set(labels)
        # check label length
        if len(unique_labels) != self.num_labels:
            self.system_panic(
                f"num_labels {self.num_labels} need to align with dataset, actual data {len(unique_labels)}!", fl_ctx
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        self.labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

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
        
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        accelerator_log_kwargs = {}

        
        accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps, **accelerator_log_kwargs)

        # set local tensorboard writer for local validation score of global model
        self.writer = SummaryWriter(app_dir)

        # set the training-related contexts, this is task-specific
        # get data from csv files
        self.log_info(fl_ctx, f"Reading data from {self.data_path}")
        raw_datasets = self.load_data()

        # get labels from data
        #self.get_labels(df_train)

        # initialize model
        self.log_info(
            fl_ctx,
            f"Creating model {self.model_path}",
        )
        
        self.model = XLMRobertaModel(model_path=self.model_path)
        model = self.model.model
        tokenizer = self.model.tokenizer
        max_seq_length = tokenizer.model_max_length
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        padding = False
        
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                #num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on every text in dataset",
            )
            
         # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    #num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )

        
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["dev"]

        # Data collator
        # This one will take care of randomly masking the tokens.
        # TODO: Set the MLM probability and batch sizes as client arguments
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=self.args.mlm_probability)

        # DataLoaders creation:
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=self.args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=self.args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=self.args.max_train_steps
            if overrode_max_train_steps
            else self.args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        self.model.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        # if accelerator.distributed_type == DistributedType.TPU:
        #     self.model.tie_weights()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = self.args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        # TODO: Enable tracking
        # if self.args.with_tracking:
        #     experiment_config = vars(self.args)
        #     # TensorBoard cannot log Enums, need the raw value
        #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        #     accelerator.init_trackers("mlm_no_trainer", experiment_config)

        
        self.total_batch_size = self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps
        self.accelerator = accelerator
        
    def local_train(
        self,
        fl_ctx,
        train_loader,
        abort_signal: Signal,
    ):
        """Typical training logic
        Total local epochs: self.aggregation_epochs
        Load data pairs from train_loader
        Compute loss with self.model
        Update model
        """
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.model.train()
            
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch_data)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    # if self.args.with_tracking:
                    #     total_loss += loss.detach().float()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    gathered_loss = self.accelerator.gather_for_metrics(loss.repeat(self.args.per_device_eval_batch_size))
                    
                    current_step = epoch_len * self.epoch_global + i
                    self.writer.add_scalar("train_loss", gathered_loss.mean().item(), current_step)
                    
                # # Checks if the accelerator has performed an optimization step behind the scenes
                # if self.accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     completed_steps += 1

                # if isinstance(checkpointing_steps, int):
                #     if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                #         output_dir = f"step_{completed_steps}"
                #         if args.output_dir is not None:
                #             output_dir = os.path.join(args.output_dir, output_dir)
                #         accelerator.save_state(output_dir)

                


    def local_valid(
        self,
        valid_loader,
        abort_signal: Signal,
        tb_id_pre=None,
        record_epoch=None,
    ):
        """Typical validation logic
        Load data pairs from train_loader
        Compute outputs with model
        Compute evaluation metric with self.valid_metric
        Add score to tensorboard record with specified id
        """
        self.model.eval()
        losses = []
        
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)

            loss = outputs.loss
            losses.append(self.accelerator.gather_for_metrics(loss.repeat(self.args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
            
        #     total_acc_val, total_loss_val, val_total = 0, 0, 0
        #     val_y_pred, val_y_true = [], []
        #     for val_data, val_label in valid_loader:
        #         if abort_signal.triggered:
        #             return make_reply(ReturnCode.TASK_ABORTED)
        #         val_label = val_label.to(self.device)
        #         val_total += val_label.shape[0]
        #         mask = val_data["attention_mask"].squeeze(1).to(self.device)
        #         input_id = val_data["input_ids"].squeeze(1).to(self.device)
        #         # Inference
        #         loss, logits = self.model(input_id, mask, val_label)
        #         # Add items for metric computation
        #         for i in range(logits.shape[0]):
        #             # remove pad tokens
        #             logits_clean = logits[i][val_label[i] != self.ignore_token]
        #             label_clean = val_label[i][val_label[i] != self.ignore_token]
        #             # calcluate acc and store prediciton and true labels
        #             predictions = logits_clean.argmax(dim=1)
        #             acc = (predictions == label_clean).float().mean()
        #             total_acc_val += acc.item()
        #             val_y_pred.append([self.ids_to_labels[x.item()] for x in predictions])
        #             val_y_true.append([self.ids_to_labels[x.item()] for x in label_clean])
        #     # compute metric
        #     metric_dict = classification_report(y_true=val_y_true, y_pred=val_y_pred, output_dict=True, zero_division=0)
        #     # tensorboard record id prefix, add to record if provided
        #     if tb_id_pre:
        #         self.writer.add_scalar(tb_id_pre + "_precision", metric_dict["macro avg"]["precision"], record_epoch)
        #         self.writer.add_scalar(tb_id_pre + "_recall", metric_dict["macro avg"]["recall"], record_epoch)
        #         self.writer.add_scalar(tb_id_pre + "_f1-score", metric_dict["macro avg"]["f1-score"], record_epoch)
        return perplexity

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
        epoch_len = len(self.train_dataloader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_dataloader,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

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

        # flush the tb writer
        self.writer.flush()

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Typical validation task pipeline
        Get global model weights (potentially with HE)
        Validation on local data
        Return validation F-1 score
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
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_metric = self.local_valid(
                self.eval_dataloader,
                abort_signal,
                tb_id_pre="val_global",
                record_epoch=self.epoch_global,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_f1_global_model ({model_owner}): {global_metric:.4f}")
            # validation metrics will be averaged with weights at server end for best model record
            metric_dxo = DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_metric},
                meta={},
            )
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.eval_dataloader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
