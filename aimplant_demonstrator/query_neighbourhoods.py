import argparse
from collections import defaultdict
from pathlib import Path
import json
import csv
import os
from functools import partial
from typing import Optional
import multiprocessing
import pickle
import json

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import trange, tqdm
from datasets import load_dataset
import h5py
import numpy as np
from sqlitedict import SqliteDict
import lancedb
#from lancedb.exceptions import LanceError

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
    parser.add_argument('test_data',
                        help="Path to text file with data to generate vectors for",
                        type=Path)
    parser.add_argument('vector_database',
                        help=("Path to directory of the lancedb database. Apart from the directory for the "
                             "database, there should be a JSON file in the parent directory with the same base "
                             "name as the directory with the `.json` suffix"),
                        type=Path)
    parser.add_argument('--target-positive', 
                        help=("If supplied, will be used to tag words as "
                              "\"target positive\" for the evaluation"), nargs="+",type=Path)
    parser.add_argument('--known-positive', 
                        help=("If supplied, will be used to tag words as \"known positive\", "
                              "which can be used downstreams to filter out in evaluation "
                              "(these terms are positive in the training data, but should "
                              "not contribute in any way to the perfomance on the "
                              "\"target positive\" words)"), nargs="+",type=Path)
    parser.add_argument('--stop-list', 
                        help="If supplied, will be used as a filter and skip any words in this list", 
                        nargs='+',type=Path)
    parser.add_argument('--output_dir',
                        help="Path to directory to save output files",
                        type=Path)
    parser.add_argument('--num-dl-workers', help="Number of processes to use when loading dataset.", type=int, default=0)
    parser.add_argument('--batch-size', help="Size of batches.", type=int, default=2)
    parser.add_argument('--query_size', 
                        help="The data is collected in chunks before being written to disk. This decides the size of those chunks", 
                        type=int, 
                        default=128)
    parser.add_argument('--n-neighbours', 
                        help="Number of neighbours to retrieve from the vector database for each query word", 
                        type=int, 
                        default=60)
    parser.add_argument('--metric', help="The metric to use for the vector database index", type=str, default='cosine')
    args = parser.parse_args()
    
    config = load_config()
    config.training_args.eval_samples = None  # Evaluate on full dev/test set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_directory = args.test_data.parent
    extension = 'text'

    cache_dir = data_directory / "hf_cache"
    
    model = load_model_from_checkpoint(args.model_path)
    model.to(device)
    dataset = load_dataset(extension, data_files={'data': str(args.test_data)}, cache_dir=cache_dir)['data']

    tokenized_dataset = model.tokenize_and_group_data(dataset, dataset.column_names)
    #collator_fn = DataCollatorWithPadding(tokenizer=model.tokenizer)
    collator_fn = partial(collate_batch, tokenizer=model.tokenizer)
    dataloader = DataLoader(tokenized_dataset, num_workers=args.num_dl_workers, batch_size=args.batch_size, collate_fn=collator_fn)
    
    database_metadata_path = args.vector_database.with_suffix('.json')
    with open(database_metadata_path) as fp:
        database_metadata = json.load(fp)
    aggregation = database_metadata['aggregation']
    layers = [int(l) for l in database_metadata['layers']]
    table_name = database_metadata.get('table_name', 'words')
    stop_list = set()
    if args.stop_list is not None:
        for stop_list_path in args.stop_list:
            with open(stop_list_path) as fp:
                stop_list = stop_list.union(w for line in fp for w in line.strip().lower().split())
    elif 'stop_words' in database_metadata:
        stop_list = set(database_metadata['stop_words'])
        
    target_positive_words = set()
    if args.target_positive is not None:
        for positive_words_path in args.target_positive:
            with open(positive_words_path) as fp:
                target_positive_words = target_positive_words.union(w for line in fp for w in line.strip().lower().split())
    elif 'positive_words' in database_metadata:
        target_positive_words = set(database_metadata['positive_words'])
    
    known_positive_words = set()
    if args.known_positive is not None:
        for known_positive_path in args.known_positive:
            with open(known_positive_path) as fp:
                known_positive_words = known_positive_words.union(w for line in fp for w in line.strip().lower().split())
    elif 'known_positive_words' in database_metadata:
        known_positive_words = set(database_metadata['known_positive_words'])

    output_dir = args.vector_database.with_name(f"{args.vector_database.stem}-{args.test_data.stem}-{args.metric}-{args.n_neighbours}-neighbourhoods")
    output_dir.mkdir(parents=True, exist_ok=True)
        # We need to check the model format for the federated training
    
    db = lancedb.connect(args.vector_database)
    table = db.open_table(table_name)
    print("Creating index...")
    try:
        table.create_index(
            metric=args.metric,
            vector_column_name="vector",
            replace=True)
    except RuntimeError as e:
        print("Index already exists, skipping index creation.")
        
    print("Index created.")
    chunk_index = 0

    word_class_mappings = { "negative": 0, "target_positive": 1, "stop_word": 2, "known_positive": 3}

    with torch.inference_mode():
        model.eval()
        for i, batch in enumerate(tqdm(dataloader)):
            predictions = model(input_ids=batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device),
                                #output_attentions=True, 
                                output_hidden_states=True)
            
            # Hidden states are lists of the layer activations
            hidden_states = predictions["hidden_states"]
            
            # Here
            layer_vectors = [hidden_states[layer_idx].cpu().numpy() for layer_idx in layers]
            # We stack so that the last axis is the layer axis. 
            # This means that if we resize, we'll get the correct order of vectors
            stacked_states = np.stack(layer_vectors, axis=-1)
            if aggregation == 'mean':
                aggregated_vectors = np.mean(stacked_states, axis=-1)
            elif aggregation == 'sum':
                aggregated_vectors = np.sum(stacked_states, axis=-1)
            elif aggregation == 'concatenate':  
                # Instead of using np.concatenate, we should probably reshape here. The states will have shape
                # (batch_size, seq_len, d_model, n_layers, ), we want to reshape so we have 
                # (batch_size, seq_len, d_model * d_model)
                aggregated_vectors = np.reshape(stacked_states, (stacked_states.shape[0], stacked_states.shape[1], -1))
            elif aggregation == 'none':
                #aggregated_vectors = np.stack(layer_vectors, axis=0)
                aggregated_vectors = stacked_states  # Lance want the multivectors as a list of vectors
            else:
                raise RuntimeError(f"Unknown aggregation method {args.aggregation}")
            
            query_word_indices = []
            query_vectors = []
            all_words = []

            for example_idx, (words, word_groups) in enumerate(zip(batch["words"], batch["word_token_groups"])):
                example_states = aggregated_vectors[0]
                for (word, word_group) in zip(words, word_groups):

                    word = word.lower()
                    if word in stop_list:
                        word_class = word_class_mappings["stop_word"]
                    elif word in target_positive_words:
                        word_class = word_class_mappings["target_positive"]
                    elif word in known_positive_words:
                        word_class = word_class_mappings["known_positive"]
                    else:
                        word_class = word_class_mappings["negative"]
                    
                    word_idx = len(all_words)
                    all_words.append((word, word_class))
                    if word not in stop_list:
                        word_vectors = example_states[word_group]
                        
                        query_vector = word_vectors.mean(axis=0)
                        query_word_indices.append(word_idx)
                        query_vectors.append(query_vector)
                        
                        if len(query_word_indices) >= args.query_size:
                            neighbourhoods = query_database(table, query_word_indices, query_vectors, all_words, k=args.n_neighbours, metric=args.metric)
                            with open(output_dir / f"neighbour_chunks_{chunk_index:02}.pkl", 'wb') as fp:
                                output = {"neighbourhoods": neighbourhoods,
                                          "class_mapping": word_class_mappings}
                                pickle.dump(output, fp)
                            chunk_index += 1
                            #print(neighbourhoods)
                            query_word_indices.clear()
                            query_vectors.clear()
                            all_words.clear()
        neighbourhoods = query_database(table, query_word_indices, query_vectors, all_words, k=args.n_neighbours, metric=args.metric)
        with open(output_dir / f"neighbour_chunks_{chunk_index:02}.pkl", 'wb') as fp:
            pickle.dump(neighbourhoods, fp)
        

                    
def query_database(table, query_word_indices, query_vectors, all_words, k=20, metric='cosine'):
    batches = table.search(query_vectors).limit(k).distance_type(metric).to_batches()
    # We will produce a list of all words like the one in `all_words`, but populate the
    # words which are relevant (not stop-words) with their neighbours. We do this by first
    # queriying the neighbours of the relevant words and record their index.
    # We will later traverse all words and populate the ones which have neighbours
    # with the response from the query.
    relevant_neighbourhood = defaultdict(list)
    
    for batch in batches:
        for qi, word, word_class, distance in zip(batch['query_index'], batch['word'], batch['label'], batch['_distance']):
            query_word_index = query_word_indices[qi]
            record = (float(distance), str(word), int(word_class))
            relevant_neighbourhood[query_word_index].append(record)
    
    collated_neighbourhood = []
    for i, word in enumerate(all_words):
        try:
            neighbours = relevant_neighbourhood[i]
        except KeyError:
            neighbours = []
        collated_neighbourhood.append((word, neighbours))
    return collated_neighbourhood


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()