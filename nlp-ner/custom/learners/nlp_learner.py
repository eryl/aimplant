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

import os

import numpy as np
import pandas as pd
import torch
from custom.models.nlp_models import BertModel, GPTModel, XLMRobertaModel
from custom.utils.data_sequence import DataSequence
from seqeval.metrics import classification_report
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NLPLearner(Learner):
    def __init__(
        self,
        data_path: str,
        model_path: str,
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

    def load_data(self):
        df_train = pd.read_csv(os.path.join(self.data_path, self.client_id + "_train.csv"))
        df_valid = pd.read_csv(os.path.join(self.data_path, self.client_id + "_val.csv"))
        return df_train, df_valid

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

        # set local tensorboard writer for local validation score of global model
        self.writer = SummaryWriter(app_dir)

        # set the training-related contexts, this is task-specific
        # get data from csv files
        self.log_info(fl_ctx, f"Reading data from {self.data_path}")
        df_train, df_valid = self.load_data()

        # get labels from data
        self.get_labels(df_train)

        # initialize model
        self.log_info(
            fl_ctx,
            f"Creating model {self.model_path}",
        )
        if self.model_path == "bert-base-uncased":
            self.model = BertModel(model_name=self.model_path, num_labels=self.num_labels)
        elif self.model_path == "gpt2":
            self.model = GPTModel(model_name=self.model_path, num_labels=self.num_labels)
        elif self.model == 'xlmroberta_large':
            self.model = XLMRobertaModel(model_name=self.model_path)
        else:
            self.system_panic(f"Model {self.model} not supported!", fl_ctx)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        #tokenizer = self.model.tokenizer

         # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
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

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        
        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        
        self.train_loader = DataLoader(train_dataset, num_workers=2, batch_size=self.bs, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, num_workers=2, batch_size=self.bs, shuffle=False)
        self.log_info(
            fl_ctx,
            f"Training Size: {len(self.train_loader.dataset)}, Validation Size: {len(self.valid_loader.dataset)}",
        )
        # Set the training-related context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

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

                mask = batch_data[0]["attention_mask"].squeeze(1).to(self.device)
                input_id = batch_data[0]["input_ids"].squeeze(1).to(self.device)
                train_label = batch_data[1].to(self.device)

                # optimize
                loss, logits = self.model(input_id, mask, train_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_step = epoch_len * self.epoch_global + i
                self.writer.add_scalar("train_loss", loss.item(), current_step)

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
        with torch.no_grad():
            total_acc_val, total_loss_val, val_total = 0, 0, 0
            val_y_pred, val_y_true = [], []
            for val_data, val_label in valid_loader:
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                val_label = val_label.to(self.device)
                val_total += val_label.shape[0]
                mask = val_data["attention_mask"].squeeze(1).to(self.device)
                input_id = val_data["input_ids"].squeeze(1).to(self.device)
                # Inference
                loss, logits = self.model(input_id, mask, val_label)
                # Add items for metric computation
                for i in range(logits.shape[0]):
                    # remove pad tokens
                    logits_clean = logits[i][val_label[i] != self.ignore_token]
                    label_clean = val_label[i][val_label[i] != self.ignore_token]
                    # calcluate acc and store prediciton and true labels
                    predictions = logits_clean.argmax(dim=1)
                    acc = (predictions == label_clean).float().mean()
                    total_acc_val += acc.item()
                    val_y_pred.append([self.ids_to_labels[x.item()] for x in predictions])
                    val_y_true.append([self.ids_to_labels[x.item()] for x in label_clean])
            # compute metric
            metric_dict = classification_report(y_true=val_y_true, y_pred=val_y_pred, output_dict=True, zero_division=0)
            # tensorboard record id prefix, add to record if provided
            if tb_id_pre:
                self.writer.add_scalar(tb_id_pre + "_precision", metric_dict["macro avg"]["precision"], record_epoch)
                self.writer.add_scalar(tb_id_pre + "_recall", metric_dict["macro avg"]["recall"], record_epoch)
                self.writer.add_scalar(tb_id_pre + "_f1-score", metric_dict["macro avg"]["f1-score"], record_epoch)
        return metric_dict["macro avg"]["f1-score"]

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
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
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
                self.valid_loader,
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
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.valid_loader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
