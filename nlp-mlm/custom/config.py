from dataclasses import dataclass
from typing import Optional, Literal, Union
import json

from peft import LoraConfig, TaskType

@dataclass
class TrainingArgs:
    mlm_probability: float = 0.1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    weight_decay: float = 1e-6
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 3
    lr_scheduler_type: Literal["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] = 'linear'
    num_warmup_steps: int = 0
    checkpointing_steps: Optional[Union[Literal['epoch'], int]] = None
    aggregation_epochs: int = 1


@dataclass
class FHConfig:
    training_args: TrainingArgs
    lora_config: LoraConfig


def get_config(config_path: str) -> FHConfig:
    with open(config_path) as fp:
        config_dict = json.load(fp)
        
        trainings_args = config_dict['training_args']
        lora_config = config_dict['lora_config']
        # As a hack, we manually parse the things which aren't basic types at this moment.
        # There's likely more elegant ways of doing this, but for now 
        # we're sticking with ones which are secure. This assumes that the task_type 
        # field in the JSON corresponds to the enum attribute of TaskType
        lora_config['task_type'] = getattr(TaskType, lora_config['task_type'])
        config = FHConfig(training_args=TrainingArgs(**config_dict['training_args']),
                          lora_config = LoraConfig(**config_dict['lora_config']))
        
        return config
    