import time

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext

class TQDMLogger: 
    """File-like class redirecting tqdm progress bar to given logging logger."""
    def __init__(self, component: FLComponent, context: FLContext, update_interval: float=60):
        self.component = component
        self.context = context
        self.last_message_at = time.time() - update_interval # We make the first call to write always output
        self.update_interval = update_interval
 
    def write(self, msg: str) -> None:
        if (time.time() - self.last_message_at) > self.update_interval:
            self.component.log_info(self.context, msg.lstrip('\r'), fire_event=True)
            self.last_message_at = time.time()

    def flush(self) -> None:
        pass    
    