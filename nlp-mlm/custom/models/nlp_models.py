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

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, TaskType, get_peft_model

class XLMRobertaModel(torch.nn.Module):
    def __init__(self, model_path):
        super(XLMRobertaModel, self).__init__()
        self.model_name = model_path
        self.base_model = AutoModelForMaskedLM.from_pretrained(
            self.model_name#, output_attentions=False, output_hidden_states=False
        )
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, 
                                 inference_mode=False, 
                                 r=8, 
                                 lora_alpha=32, 
                                 lora_dropout=0.1)
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output
    
    def parameters(self, recurse = True):
        return super().parameters(recurse)
    
    def state_dict(self):
        # We filter to only include the LoRA weights
        lora_weights = {k: v for k,v in self.model.state_dict().items() if 'lora' in k}
        return lora_weights
    
    def load_state_dict(self, state_dict):
        # Since we've filtered the state dict, this will only have the lora weights so we disable strict mode
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        pass # just so I can place a breakpoint on this line to inspect the results from the previous line
    
    def initialize(self):
        pass