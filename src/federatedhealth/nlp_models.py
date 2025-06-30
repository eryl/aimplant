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
import datetime
import math
import os
from pathlib import Path
from itertools import chain
import time
from typing import Optional, Literal, Union, TypeVar, Type
import importlib

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, TaskType, get_peft_model

from federatedhealth.config import TrainingArgs, load_config

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


class XLMRobertaModel(torch.nn.Module):
    def __init__(self):
        # Keep in mind that this initializer will run both on
        # the server and the clients. We should not do things
        # dependent on the client training here (such as
        # setting up dataloaders), instead do that in initialize()
        
        super(XLMRobertaModel, self).__init__()
        
        self.config = load_config()
                
        self.model_name = self.config.model_path
        self.base_model = AutoModelForMaskedLM.from_pretrained(
            self.model_name#, output_attentions=False, output_hidden_states=False
        )
        
        for name, param in self.base_model.named_parameters():
            # We actually wish to train the bias parameters, 
            # so only set non-bias parameters to not require grad
            if 'bias' not in name:
                param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.training_args = self.config.training_args
        self.peft_config = self.config.lora_config
        # Set up LoRA
        self.model = get_peft_model(self.base_model, self.peft_config)
        #self.model.print_trainable_parameters()
        #self.model = self.base_model
        self.current_step = 0
        self.aggregation_epochs = self.training_args.aggregation_epochs
        

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output
    
    def parameters(self, recurse = True):
        return super().parameters(recurse)
    
    def state_dict(self):
        # We filter to only include the LoRA weights
        state_dict = {k: v for k,v in self.model.state_dict().items() if 'lora' in k or 'bias' in k}
        return state_dict
    
    def load_state_dict(self, state_dict):
        # Since we've filtered the state dict, this will only have the lora weights so we disable strict mode
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        pass # just so I can place a breakpoint on this line to inspect the results from the previous line
    
    def load_data(self, train_dataset_path, dev_dataset_path, test_dataset_path):
        #train_dataset_path = os.path.join(self.data_path, self.client_id + "_train.txt")
        #train_dataset = load_dataset(train_dataset_path)
        #dev_dataset_path = os.path.join(self.data_path, self.client_id + "_dev.txt")
        #test_dataset_path = os.path.join(self.data_path, self.client_id + "_test.txt")
        #dev_dataset = load_dataset(dev_dataset_path)
        data_files = {'train': train_dataset_path, 'dev': dev_dataset_path, 'test': test_dataset_path}
        extension = 'text'
        
        # Don't cache things in the default place, assume there is space in the dataset directory
        cache_dir = os.path.join(os.path.dirname(train_dataset_path), "hf_cache")
        dataset = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
        return dataset
    
    def initialize(self, app_dir, train_dataset_path, dev_dataset_path, test_dataset_path):
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        accelerator_log_kwargs = {}
        
        with open(os.path.join(app_dir, "trace.txt"), 'a') as fp:
            timestamp = datetime.datetime.now()
            fp.write(f"XLMRobertaModel.initialize: {timestamp}\n")
        
        #training_batch_size = os.environ.get("FH_TRAIN_BATCH_SIZE", self.training_args.per_device_train_batch_size)
        #eval_batch_size = os.environ.get("FH_EVAL_BATCH_SIZE", self.training_args.per_device_eval_batch_size)
        
        training_batch_size = self.training_args.per_device_train_batch_size
        eval_batch_size = self.training_args.per_device_eval_batch_size
        
        gradient_accumulation_steps = self.training_args.optimization_batch_size // training_batch_size
        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, **accelerator_log_kwargs)

        # set local tensorboard writer for local validation score of global model
        self.writer = SummaryWriter(app_dir)
        
        raw_datasets = self.load_data(train_dataset_path, dev_dataset_path, test_dataset_path)
        
        max_seq_length = self.tokenizer.model_max_length
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        padding = False
        
        tokenizer = self.tokenizer
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        # Where are the cached versions saved?
        with self.accelerator.main_process_first():
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
        
        with self.accelerator.main_process_first():
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
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=self.training_args.mlm_probability)


        # DataLoaders creation:
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        include_params = ['lora', 'bias']
        no_decay = ["bias", "LayerNorm.weight"]
        no_decay_params = {"params": [], "weight_decay": 0.0}
        decay_params = {"params": [], "weight_decay": self.training_args.weight_decay}
        
        for n, p in self.model.named_parameters():
            if include_params is not None and any(incl in n for incl in include_params):
                if any(nd in n for nd in no_decay):
                    no_decay_params["params"].append(p)
                else:
                    decay_params["params"].append(p)
                
        optimizer_grouped_parameters = [no_decay_params, decay_params]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)
        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        
        if self.training_args.max_train_steps is None:
            self.training_args.max_train_steps = self.training_args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.training_args.num_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.training_args.max_train_steps
            if overrode_max_train_steps
            else self.training_args.max_train_steps * self.accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        # if accelerator.distributed_type == DistributedType.TPU:
        #     self.model.tie_weights()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.training_args.max_train_steps = self.training_args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.training_args.num_train_epochs = math.ceil(self.training_args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = self.training_args.checkpointing_steps
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

        self.total_batch_size = self.training_args.per_device_train_batch_size * self.accelerator.num_processes * gradient_accumulation_steps
    
    
    def fit_batch(self, batch_data, current_step=None):
            
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

            gathered_loss = self.accelerator.gather_for_metrics(loss.repeat(self.training_args.per_device_eval_batch_size))
            self.current_step += 1
            if current_step is None:
                current_step = self.current_step
            #current_step = epoch_len * self.epoch_global + i
            self.writer.add_scalar("train_loss", gathered_loss.mean().item(), current_step)
            self.writer.flush()

    def train_for_epochs(self, epochs=None):
        """Typical training logic
        Total local epochs: self.aggregation_epochs
        Load data pairs from train_loader
        Compute loss with self.model
        Update model
        """
        if epochs is None:
            epochs = self.aggregation_epochs
            
        for epoch in range(epochs):
            self.model.train()
            for i, batch_data in enumerate(self.train_dataloader):
                self.fit_batch(batch_data)
                
                    
                # # Checks if the accelerator has performed an optimization step behind the scenes
                # if self.accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     completed_steps += 1
                # TODO make the local training also save the model
                # if isinstance(checkpointing_steps, int):
                #     if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                #         output_dir = f"step_{completed_steps}"
                #         if args.output_dir is not None:
                #             output_dir = os.path.join(args.output_dir, output_dir)
                #         accelerator.save_state(output_dir)


    def local_valid(self, current_step=None):
        self.model.eval()
        losses = []
        
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)

            loss = outputs.loss
            losses.append(self.accelerator.gather_for_metrics(loss.repeat(self.training_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        if current_step is None:
            current_step = self.current_step
        try:
            eval_loss = torch.mean(losses)
            self.writer.add_scalar('dev_loss', eval_loss.item(), current_step)
            perplexity = math.exp(eval_loss)
            self.writer.add_scalar('perplexity', perplexity, current_step)
        except OverflowError:
            perplexity = float("inf")
        
        return perplexity