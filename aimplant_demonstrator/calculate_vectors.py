import argparse
from pathlib import Path
import json
import csv
import os
from functools import partial

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
    parser = argparse.ArgumentParser(description="Create vector database for trained XLMRoberta models on the test data")
    parser.add_argument('model_path',
                        help="Path to the model checkpoint to use for generating vectors",
                        type=Path)
    parser.add_argument('data',
                        help="Path to text file with data to generate vectors for",
                        type=Path)
    parser.add_argument('output_dir',
                        help="Path to directory to save output files",
                        type=Path)
    parser.add_argument('--num-dl-workers', help="Number of processes to use when loading dataset.", type=int, default=8)
    parser.add_argument('--batch-size', help="Size of batches.", type=int, default=8)
    parser.add_argument('--layer-states', help="What layers to get activations from.", type=int, default=[-4, -3, -2, -1])
    args = parser.parse_args()
    
    config = load_config()
    config.training_args.eval_samples = None  # Evaluate on full dev/test set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_directory = args.data.parent
    extension = 'text'

    cache_dir = data_directory / "hf_cache"
    
    output_dir = args.output_dir if args.output_dir else args.model_path.parent / "vector_database"
    output_dir.mkdir(parents=True, exist_ok=True)
        # We need to check the model format for the federated training
    model = load_model_from_checkpoint(args.model_path)
    model.to(device)
    dataset = load_dataset(extension, data_files={'data': str(args.data)}, cache_dir=cache_dir)['data']

    tokenized_dataset = model.tokenize_and_group_data(dataset, dataset.column_names)
    #collator_fn = DataCollatorWithPadding(tokenizer=model.tokenizer)
    collator_fn = partial(collate_batch, tokenizer=model.tokenizer)
    dataloader = DataLoader(tokenized_dataset, num_workers=args.num_dl_workers, batch_size=args.batch_size, collate_fn=collator_fn)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    word_vectors_file = args.output_dir / 'vectors.h5'
    word_sums_file = args.output_dir / 'vector_sums.sqlite'

    with torch.inference_mode(), h5py.File(word_vectors_file, 'w') as store, SqliteDict(word_sums_file, autocommit=False) as db:
        model.eval()
        vector_sums = dict()
        vector_chunk = []
        words_chunk = []
        chunk_size = 128

        for batch in tqdm(dataloader):
            predictions = model(input_ids=batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device),
                                #output_attentions=True, 
                                output_hidden_states=True)
            
            hidden_states = predictions["hidden_states"]
            
            for example_idx, (words, word_groups) in enumerate(zip(batch["words"], batch["word_token_groups"])):
                for word_idx, (word, word_group) in enumerate(zip(words, word_groups)):
                    vector_stack = []
                    for layer_idx in args.layer_states:
                        layer_states = hidden_states[layer_idx]
                        word_vectors = layer_states[example_idx, word_group]
                        word_vector = word_vectors.mean(dim=0)
                        vector_stack.append(word_vector)
                    stacked = torch.stack(vector_stack).to(torch.float16).cpu().numpy()
                    vector_chunk.append(stacked)
                    words_chunk.append(word)
                    if len(vector_chunk) >= chunk_size:
                        store_data(vector_chunk, words_chunk, store, chunk_size=chunk_size)
                        vector_chunk.clear()
                        words_chunk.clear()

                    if word not in vector_sums:
                        vector_sums[word] = (1, stacked)
                    else:
                        prev_count, prev_sum = vector_sums[word]
                        vector_sums[word] = (prev_count + 1, prev_sum + stacked)
                    
                    if len(vector_sums) > chunk_size:
                        sync_vector_sums(vector_sums, db)
                        vector_sums.clear()
                    

        store_data(vector_chunk, words_chunk, store, chunk_size=chunk_size)
        sync_vector_sums(vector_sums, db)
                        
        

def sync_vector_sums(vector_sums, db):
    for word, (count, vectors) in vector_sums.items():
        if word in db:
            previous_count, previous_vectors = db[word]
            count += previous_count
            vectors += previous_vectors
        db[word] = (count, vectors)
    db.commit()

def store_data(vector_chunk, word_chunk, store, chunk_size=128):
    stacked_vector_chunk = np.stack(vector_chunk)
    n_vectors, *vector_shapes = stacked_vector_chunk.shape
    if 'vectors' not in store:
        store.create_dataset('vectors', 
                                data=stacked_vector_chunk, 
                                chunks=(chunk_size, *vector_shapes), 
                                maxshape=(None, *vector_shapes),
                                compression='gzip',
                                compression_opts=9)
        store.create_dataset('words', data=word_chunk, dtype=h5py.string_dtype(), maxshape=(None,))
    else:
        current_size = store['vectors'].shape[0]
        new_size = current_size + n_vectors
        store['vectors'].resize(new_size, axis=0)
        store['vectors'][current_size:new_size] = stacked_vector_chunk
        store['words'].resize(new_size, axis=0)
        store['words'][current_size:new_size] = word_chunk


                
            
    
    
if __name__ == '__main__':
    main()