from dataclasses import dataclass, fields
from typing import Optional, Literal, Union, Dict
import json

from peft import LoraConfig, TaskType

import os
import shutil
from pathlib import Path
from importlib.resources import files

def get_user_config_path():
    return Path.home() / ".federatedhealth" / "config.json"

def ensure_config_exists():
    config_path = get_user_config_path()
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Load default from package
        default_path = files("federatedhealth").joinpath("default_config.json")
        shutil.copy(default_path, config_path)
    return config_path


@dataclass
class TrainingArgs:
    mlm_probability: float = 0.1
    optimization_batch_size: int = 32
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    disable_tqdm: Optional[bool] = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 3
    lr_scheduler_type: Literal["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] = 'linear'
    num_warmup_steps: int = 0
    checkpointing_steps: Optional[Union[Literal['epoch'], int]] = None
    aggregation_epochs: int = 1
    num_train_workers: int = 16
    num_eval_workers: int = 16
    train_samples: Optional[int] = None  # If given, training will only use this many samples
    eval_samples: Optional[int] = None  # If steps are given, evaluation will only use these many samples

    def update(self, updates: Dict):
        for f in fields(self):
            if f.name in updates:
                setattr(self, f.name, updates[f.name])
    
@dataclass
class DataConfig:
    training_data: str
    dev_data: str
    test_data: str

@dataclass
class FHConfig:
    training_args: TrainingArgs
    lora_config: LoraConfig
    data_config: DataConfig
    model_path: str
    
    

def load_config(override_path: Optional[str]=None) -> FHConfig:
    config_path = ensure_config_exists()
    with open(config_path) as f:
        config_dict = json.load(f)
        trainings_args = config_dict['training_args']
        lora_config = config_dict['lora_config']
        data_config = config_dict['data_config']
        
        if override_path is not None and os.path.exists(override_path):
            with open(override_path) as fp:
                override_config = json.load(fp)
                override_training_args = override_config.get('training_args')
                if override_training_args is not None:
                    trainings_args.update(override_training_args)
                override_lora_args = override_config.get('lora_config')
                if override_lora_args is not None:
                    lora_config.update(override_lora_args)
            
            # We did a check to make sure config files exist. This breaks things server-side, where configs do not have to exist.
            # all_exists = True
            # for path_attr in fields(data_config):
            #     path = getattr(data_config, path_attr.name)
            #     if not os.path.exists(path):
            #         print(f"Path for data_config.{path_attr.name} ({path}) does not exist!")
            #         all_exists = False
            
            # if not all_exists:
            #     raise RuntimeError(f"Dataset paths for config file {config_path} are incorrect.")
            
            # if not os.path.exists(model_path):
            #     raise RuntimeError(f"Model path {model_path} from config file {config_path} does not exist.")
    
    # As a hack, we manually parse the things which aren't basic types at this moment.
    # There's likely more elegant ways of doing this, but for now 
    # we're sticking with ones which are secure. This assumes that the task_type 
    # field in the JSON corresponds to the enum attribute of TaskType
    lora_config['task_type'] = getattr(TaskType, lora_config['task_type'])
    training_args = TrainingArgs(**trainings_args)
    lora_config = LoraConfig(**lora_config)
    data_config = DataConfig(**data_config)
    model_path = config_dict['model_path']
        
    config = FHConfig(model_path=model_path, training_args=training_args,
                        lora_config=lora_config,
                        data_config=data_config)
    return config
    