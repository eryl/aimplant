import json
import os
import re
from collections import OrderedDict
from typing import Dict
from pathlib import Path
import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.fuel.utils import fobs


from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

class EveryEpochPersistor(PTFileModelPersistor):
    def save_model(self, ml: ModelLearnable, fl_ctx: FLContext):
        self._get_persistence_manager(fl_ctx).update(ml)
        current_round = fl_ctx.get_prop('current_round')
        base_checkpoint_path = Path(self._ckpt_save_path)
        checkpoint_path = base_checkpoint_path.with_name(f"{base_checkpoint_path.stem}_{current_round}.pt")
        self.save_model_file(str(checkpoint_path))