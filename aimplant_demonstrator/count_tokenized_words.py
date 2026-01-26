import argparse
from pathlib import Path
import json
import csv
import os
from functools import partial
from typing import Optional
from collections import Counter
import csv

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import trange, tqdm
from datasets import load_dataset
import h5py
import numpy as np
from sqlitedict import SqliteDict

from federatedhealth.nlp_models import XLMRobertaModel, load_model_from_checkpoint
from federatedhealth.config import load_config

def collate_batch(batch, tokenizer):
    keys = batch[0].keys()
    collated = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            collated[k].append(b[k])
    padded = tokenizer.pad({"input_ids": collated["input_ids"], "special_tokens_mask": collated["special_tokens_mask"]}, return_tensors="pt")
    collated.update(padded)
    return collated

def main():
    parser = argparse.ArgumentParser(description="Create statistics over words based on the tokenizer of a model")
    parser.add_argument('model_path',
                        help="Path to the model checkpoint to use for generating vectors",
                        type=Path)
    parser.add_argument('data',
                        help="Path to text file with data to generate vectors for",
                        type=Path)
    parser.add_argument('--output_dir',
                        help="Path to directory to save output files",
                        type=Path)
    parser.add_argument('--num-dl-workers', help="Number of processes to use when loading dataset.", type=int, default=8)
    parser.add_argument('--batch-size', help="Size of batches.", type=int, default=8)
    args = parser.parse_args()
    
    config = load_config()
    config.training_args.eval_samples = None  # Evaluate on full dev/test set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_directory = args.data.parent
    extension = 'text'

    cache_dir = data_directory / "hf_cache"
    
    output_dir = args.output_dir if args.output_dir else args.model_path.parent / f"vector_database_{args.model_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
        # We need to check the model format for the federated training
    model = load_model_from_checkpoint(args.model_path)
    model.to(device)
    dataset = load_dataset(extension, data_files={'data': str(args.data)}, cache_dir=cache_dir)['data']

    tokenized_dataset = model.tokenize_and_group_data(dataset, dataset.column_names)
    #collator_fn = DataCollatorWithPadding(tokenizer=model.tokenizer)
    collator_fn = partial(collate_batch, tokenizer=model.tokenizer)
    dataloader = DataLoader(tokenized_dataset, num_workers=args.num_dl_workers, batch_size=args.batch_size, collate_fn=collator_fn)
    with torch.inference_mode():
        model.eval()
        word_counts = Counter()
        
        for batch in tqdm(dataloader):
            for example_idx, (words, word_groups) in enumerate(zip(batch["words"], batch["word_token_groups"])):
                for word_idx, (word, word_group) in enumerate(zip(words, word_groups)):
                    word = word.lower()
                    word_counts[word] += 1
    with open(output_dir / "word_frequencies.txt", 'w', encoding='utf-8', newline='') as fp:
        csv_writer = csv.DictWriter(fp, fieldnames=['term', 'count'])
        csv_writer.writeheader()
        for term, count in word_counts.most_common():
            csv_writer.writerow({'term': term, 'count': count})
                    
if __name__ == '__main__':
    main()