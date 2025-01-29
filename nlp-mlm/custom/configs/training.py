from dataclasses import dataclass
from typing import Optional, Literal, Union

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
