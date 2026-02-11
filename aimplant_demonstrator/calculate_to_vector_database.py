import argparse
from pathlib import Path
import json
import csv
import os
from functools import partial
from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import trange, tqdm
from datasets import load_dataset
import h5py
import numpy as np
from sqlitedict import SqliteDict
import lancedb
import pyarrow as pa

from federatedhealth.nlp_models import XLMRobertaModel, load_model_from_checkpoint
from federatedhealth.config import load_config

def collate_batch(batch, tokenizer):
    keys = batch[0].keys()
    collated = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            collated[k].append(b[k])
    padded = tokenizer.pad({'input_ids': collated['input_ids'], 'special_tokens_mask': collated['special_tokens_mask']}, return_tensors='pt')
    collated.update(padded)
    return collated


def main():
    parser = argparse.ArgumentParser(description='Create vector database for trained XLMRoberta models on the test data')
    parser.add_argument('model_path',
                        help='Path to the model checkpoint to use for generating vectors',
                        type=Path)
    parser.add_argument('data',
                        help='Path to text file with data to generate vectors for',
                        type=Path)
    parser.add_argument('--positive-words', help='If supplied, will be used to tag words as \'positive\', which can be used downstreams', type=Path)
    parser.add_argument('--stop-list', help='If supplied, will be used as a filter and skip any words in this list', type=Path)
    parser.add_argument('--output-dir',
                        help='Path to directory to save output files',
                        type=Path)
    parser.add_argument('--num-dl-workers', help='Number of processes to use when loading dataset.', type=int, default=8)
    parser.add_argument('--batch-size', help='Size of batches.', type=int, default=4)
    parser.add_argument('--layer-states', 
                        help='What layers to get activations from.', 
                        type=int, 
                        default=[-4, -3, -2, -1])
    parser.add_argument('--aggregation', help='Method of aggregating vectors', choices=('none', 'mean', 'sum', 'concatenate'), default='mean')
    parser.add_argument('--chunk-size', 
                        help='The data is collected in chunks before being written to disk. This decides the size of those chunks', 
                        type=int, 
                        default=10000)
    parser.add_argument('--word-table-name', help='The name of the table holding the words and vectors', default='words')
    args = parser.parse_args()
    
    config = load_config()
    config.training_args.eval_samples = None  # Evaluate on full dev/test set
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_directory = args.data.parent
    extension = 'text'

    cache_dir = data_directory / 'hf_cache'
    output_dir = args.output_dir if args.output_dir else args.model_path.parent / f'vector_database_{args.model_path.resolve().stem}'
    output_dir.mkdir(parents=True, exist_ok=True)
        # We need to check the model format for the federated training
    model = load_model_from_checkpoint(args.model_path)
    model.to(device)
    dataset = load_dataset(extension, data_files={'data': str(args.data)}, cache_dir=cache_dir)['data']

    tokenized_dataset = model.tokenize_and_group_data(dataset, dataset.column_names)
    #collator_fn = DataCollatorWithPadding(tokenizer=model.tokenizer)
    collator_fn = partial(collate_batch, tokenizer=model.tokenizer)
    dataloader = DataLoader(tokenized_dataset, num_workers=args.num_dl_workers, batch_size=args.batch_size, collate_fn=collator_fn)
    word_sums_files = {layer_idx: output_dir / f'vector_sums_layer_{layer_idx}.sqlite' for layer_idx in args.layer_states}
    #word_sums_dbs = {layer_idx: SqliteDict(f, autocommit=False) for layer_idx, f in word_sums_files.items()}

    stop_list = set()
    if args.stop_list is not None:
        with open(args.stop_list) as fp:
            stop_list = set(w for line in fp for w in line.strip().lower().split())
    
    positive_words = set()
    if args.positive_words is not None:
        with open(args.positive_words) as fp:
            positive_words = set(w for line in fp for w in line.strip().lower().split())
    

    database_path = output_dir / f'lancedb_direct'
    metadata_uri = database_path.with_suffix('.json')
    with open(metadata_uri, 'w') as fp:
        db_metadata = {'layers': args.layer_states, 
                       'aggregated_words': False,
                       'aggregation': args.aggregation,
                       'wordpiece_aggregation': 'mean',
                       'stop_words': sorted(stop_list),
                       'positive_words': sorted(positive_words),
                       'word_table_name': args.word_table_name}
        json.dump(db_metadata, fp, indent=2)

    with torch.inference_mode(), DataBaseHandler(database_path, chunk_size=args.chunk_size, word_table_name=args.word_table_name) as vectordb:
        model.eval()
        vector_sums = {l: {} for l in args.layer_states}
        
        for batch in tqdm(dataloader):
            predictions = model(input_ids=batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device),
                                #output_attentions=True, 
                                output_hidden_states=True)
            
            hidden_states = [(layer_idx, predictions['hidden_states'][layer_idx].cpu().numpy()) for layer_idx in args.layer_states]
            
            for example_idx, (words, word_groups) in enumerate(zip(batch['words'], batch['word_token_groups'])):
                for word_idx, (word, word_group) in enumerate(zip(words, word_groups)):
                    word = word.lower()
                    if word in stop_list:
                        continue
                    
                    layer_vectors = []
                    for layer_idx, layer_states in hidden_states:
                        word_vectors = layer_states[example_idx, word_group]
                        local_word_vector = word_vectors.mean(axis=0)
                        #local_word_vector = word_vector.cpu().numpy()
                        
                        layer_vectors.append(local_word_vector)
                        # layer_vector_sums = vector_sums[layer_idx]
                        # if word not in layer_vector_sums:
                        #     layer_vector_sums[word] = (1, local_word_vector)
                        # else:
                        #     prev_count, prev_sum = layer_vector_sums[word]
                        #     layer_vector_sums[word] = (prev_count + 1, prev_sum + local_word_vector)
                    
                    positive_word = word in positive_words
                    labeled_word = (positive_word, word)
                    if args.aggregation == 'mean':
                        aggregated_vectors = np.mean(layer_vectors, axis=0)
                    elif args.aggregation == 'sum':
                        aggregated_vectors = np.sum(layer_vectors, axis=0)
                    elif args.aggregation == 'concatenate':
                        aggregated_vectors = np.concatenate(layer_vectors)
                    elif args.aggregation == 'none':
                        #aggregated_vectors = np.stack(layer_vectors, axis=0)
                        aggregated_vectors = layer_vectors  # Lance want the multivectors as a list of vectors
                    else:
                        raise RuntimeError(f"Unknown aggregation method {args.aggregation}")
                    vectordb.add_data(labeled_word, aggregated_vectors)
                    
                    # for layer_idx, layer_vector_sums in vector_sums.items():
                    #     if len(layer_vector_sums) > args.chunk_size:
                    #         db = word_sums_dbs[layer_idx]
                    #         sync_vector_sums(layer_vector_sums, db)
                    #         layer_vector_sums.clear()
        
        # for layer_idx, layer_vector_sums in vector_sums.items():
        #     db = word_sums_dbs[layer_idx]
        #     sync_vector_sums(layer_vector_sums, db)
        #     layer_vector_sums.clear()
        #     db.close()

def sync_vector_sums(vector_sums, db):
    for word, (count, vectors) in vector_sums.items():
        if word in db:
            previous_count, previous_vectors = db[word]
            count += previous_count
            vectors += previous_vectors
        db[word] = (count, vectors)
    db.commit()


class DataBaseHandler:
    def __init__(self, database_path: Path, chunk_size=128, word_table_name='words'):
        self.words = []
        self.vectors = []
        self.chunk_size = chunk_size
        self.uri = database_path
        self.table = None
        self.db = lancedb.connect(self.uri)
        self.word_table_name = word_table_name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.sync_vectors()

    def add_data(self, labeled_word: tuple[int, str], vector: np.ndarray):
        self.words.append(labeled_word)
        self.vectors.append(vector)
        if len(self.words) >= self.chunk_size:
            self.sync_vectors()
    
    def sync_vectors(self):
        if len(self.words) > 0:
            if self.table is None:
                v = self.vectors[0]
                if isinstance(v, list):
                    vector_dimension = v[0].shape[0]
                    vector_type = pa.list_(pa.list_(pa.float16(), vector_dimension))
                else:
                    vector_dimension = v.shape[0]
                    vector_type = pa.list_(pa.float16(), vector_dimension)
                schema = pa.schema(
                    [
                        pa.field('id', pa.int64()),
                        pa.field('label', pa.int8()),
                        pa.field('word', pa.string()),
                        pa.field('vector', vector_type)
                    ]
                )
                self.table = self.db.create_table(self.word_table_name, schema=schema, mode='overwrite')
            data = [{'word': word, 'label': label, 'vector': vector.astype(np.float16)} for (label, word), vector in zip(self.words, self.vectors)]
            self.table.add(data, on_bad_vectors='drop')

            self.words.clear()
            self.vectors.clear()

    
def store_data(vector_chunk, word_chunk, word_class_chunk, store, chunk_size=128):
    if 'words' not in store:
        store.create_dataset('words', data=word_chunk, dtype=h5py.string_dtype(), maxshape=(None,))
        store.create_dataset('word_classes', data=np.array(word_class_chunk), maxshape=(None,))
    else:
        current_size = store['words'].shape[0]
        new_size = current_size + len(word_chunk)
        store['words'].resize(new_size, axis=0)
        store['words'][current_size:new_size] = word_chunk
        store['word_classes'].resize(new_size, axis=0)
        store['word_classes'][current_size:new_size] = np.array(word_class_chunk)

    vectors_group = store.require_group('vectors')
        
    for layer_idx, vectors in vector_chunk.items():
        stacked_vector_chunk = np.stack(vectors).astype(np.float16)
        n_vectors, *vector_shapes = stacked_vector_chunk.shape
        layer_name = f'layer:{layer_idx}'
        
        if layer_name not in vectors_group:
            layer_dataset = vectors_group.create_dataset(layer_name, 
                                    data=stacked_vector_chunk, 
                                    chunks=(chunk_size, *vector_shapes), 
                                    maxshape=(None, *vector_shapes),
                                    compression='gzip',
                                    compression_opts=9)
            layer_dataset.attrs['layer_idx'] = layer_idx
        else:
            current_size = vectors_group[layer_name].shape[0]
            new_size = current_size + n_vectors
            vectors_group[layer_name].resize(new_size, axis=0)
            vectors_group[layer_name][current_size:new_size] = stacked_vector_chunk
            


                
            
    
    
if __name__ == '__main__':
    main()