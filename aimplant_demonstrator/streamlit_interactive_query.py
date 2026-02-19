import argparse
from pathlib import Path
import json
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
import streamlit as st

from federatedhealth.nlp_models import XLMRobertaModel, load_model_from_checkpoint
from federatedhealth.config import load_config

class DummyTable:
    def search(self, query_vectors):
        return self
    
    def limit(self, k):
        return self
    
    def distance_type(self, metric):
        return self
    
    def to_batches(self):
        return []

def compute_embeddings(model, query, layers, aggregation='mean', stop_list=None):
    if stop_list is None:
        stop_list = set()
    output, input_batch = model.single_forward([query], return_input_batch=True)
    hidden_states = output["hidden_states"]
    
    # Here
    layer_vectors = [hidden_states[layer_idx].cpu().numpy() for layer_idx in layers]
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
        raise RuntimeError(f"Unknown aggregation method {aggregation}")

    query_word_indices = []
    query_vectors = []
    all_words = []

    for example_idx, (words, word_groups) in enumerate(zip(input_batch["words"], input_batch["word_token_groups"])):
        example_states = aggregated_vectors[0]
        example_words = []
        example_word_indices = []
        example_word_vectors = []
        for (word, word_group) in zip(words, word_groups):

            word = word.lower()
            
            word_idx = len(example_words)
            example_words.append(word)
            if word not in stop_list:
                word_vectors = example_states[word_group]
                
                query_vector = word_vectors.mean(axis=0)
                example_word_indices.append((example_idx, word_idx))
                example_word_vectors.append(query_vector)
        all_words.append(example_words)
        query_word_indices.extend(example_word_indices)
        query_vectors.extend(example_word_vectors)
    return all_words, query_word_indices, query_vectors

                    
def make_query(table, query_word_indices: list[tuple[int, int]], query_vectors: list[np.ndarray], all_words: list[list[str]], k=20, metric='cosine'):
    batches = table.search(query_vectors).limit(k).distance_type(metric).to_batches()
    # We will produce a list of all words like the one in `all_words`, but populate the
    # words which are relevant (not stop-words) with their neighbours. We do this by first
    # queriying the neighbours of the relevant words and record their index.
    # We will later traverse all words and populate the ones which have neighbours
    # with the response from the query.
    relevant_neighbourhood = defaultdict(list)
    
    for batch in batches:
        for qi, word, word_class, distance in zip(batch['query_index'], batch['word'], batch['label'], batch['_distance']):
            example_index, word_index = query_word_indices[qi]
            record = (float(distance), str(word), int(word_class))
            relevant_neighbourhood[(example_index, word_index)].append(record)
    
    collated_neighbourhood = []
    for example_index, words in enumerate(all_words):
        example_neighbourhood = []
        for word_index, word in enumerate(words):
            try:
                neighbours = relevant_neighbourhood[(example_index, word_index)]
            except KeyError:
                neighbours = []
            example_neighbourhood.append((word, neighbours))
        collated_neighbourhood.append(example_neighbourhood)
    return collated_neighbourhood




def main():
    
    return model, table, stop_list, layers, aggregation
            
                

parser = argparse.ArgumentParser(description="Interactive query for Aimplant Demonstrator")
parser.add_argument("model_path", type=Path, help="Path to the trained model")
parser.add_argument('vector_database',
                    help=("Path to directory of the lancedb database. Apart from the directory for the "
                            "database, there should be a JSON file in the parent directory with the same base "
                            "name as the directory with the `.json` suffix"),
                    type=Path)
parser.add_argument('--stop-list', help='If supplied, will be used as a filter and skip any words in this list', type=Path, nargs='+')
parser.add_argument('--n-neighbours', 
                    help="Number of neighbours to retrieve from the vector database for each query word", 
                    type=int, 
                    default=8)
parser.add_argument('--metric', help="The metric to use for the vector database index", type=str, default='cosine')
parser.add_argument('--table-name', help="Name of the table in the vector database to query", type=str, default='words')
parser.add_argument('--dummy-table', help=("If supplied, will not connect to the "
                    "vector database and will use a dummy table which returns no results. "
                    "Useful for testing the rest of the pipeline without needing to set up "
                    "a vector database."), action='store_true')
args = parser.parse_args()

config = load_config()
config.training_args.eval_samples = None  # Evaluate on full dev/test set
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = load_model_from_checkpoint(args.model_path)
model.to(device)
model.eval()

database_metadata_path = args.vector_database.with_suffix('.json')
with open(database_metadata_path) as fp:
    database_metadata = json.load(fp)
aggregation = database_metadata['aggregation']
layers = [int(l) for l in database_metadata['layers']]

table_name = database_metadata.get('table_name', 'words')
if args.table_name is not None:
    table_name = args.table_name

stop_list = set()
if args.stop_list is not None:
    for stop_list_path in args.stop_list:
        with open(stop_list_path) as fp:
            stop_list = stop_list.union(w for line in fp for w in line.strip().lower().split())
elif 'stop_words' in database_metadata:
    stop_list = set(database_metadata['stop_words'])


if not args.dummy_table:
    db = lancedb.connect(args.vector_database)
    table = db.open_table(table_name)
    try:
        table.create_index(
            metric=args.metric,
            vector_column_name="vector",
            replace=False)
    except RuntimeError as e:
        #print("Index already exists, skipping index creation.")
        pass 
        
else:
    table = DummyTable()

with torch.inference_mode():
    n_neighbours = st.number_input("Number of neighbours to retrieve from the vector database for each query word", min_value=1, value=args.n_neighbours)
    query_input = st.text_area("Enter a query", key="query_input")
    if len(query_input.split()) > 1:
        all_words, word_indices, word_vectors = compute_embeddings(model, query_input, layers=layers, aggregation=aggregation, stop_list=stop_list)
        query_result = make_query(table, word_indices, word_vectors, all_words, n_neighbours)
        query_result = query_result[0] # The above functions works on batches, but we have a single example
        for word, neighbours in query_result:
            st.write(word)
            if neighbours:
                df = st.dataframe([{"Neighbour": neighbour_word, "Distance": distance, "Class": word_class} for distance, neighbour_word, word_class in neighbours])
            
        
    # while True:
    #     query = input("Enter a query (or '/quit' to exit, '/help' for help): ")
    #     query_parts = query.lower().strip().split()
    #     if query_parts and query_parts[0] == '/quit':
    #         break
    #     elif query_parts and query_parts[0] == '/help':
    #         print("Available commands:")
    #         print("  /quit - Exit the interactive query session")
    #         print("  /help - Show this help message")
    #         print("  /neighbours N or /n N - Change the number of neighbours to retrieve from the vector database for each query word (default: 16)")
    #     elif query_parts and query_parts[0] in ("/neighbours", "/n"):
    #         try:
    #             n_neighbours = int(query_parts[1])
    #         except (ValueError, IndexError):
    #             print(f"Invalid number of neighbours {query_parts[1] if len(query_parts) > 1 else 'None'}. Please provide an integer.")
    #     elif query_parts:
    #         
    #     else:
    #         print("Please enter a non-empty query or a command. Type '/help' for available commands.")
