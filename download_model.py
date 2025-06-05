import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


model_name = "xlm-roberta-base"  # Replace with any model from the hub

model_path = os.environ.get('FH_MODEL_DIR', None)
if model_path is None:
    model_path = 'models'

model_path = Path(model_path)
model_path.mkdir(exist_ok=True, parents=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
model = AutoModel.from_pretrained(model_name, cache_dir=model_path)

