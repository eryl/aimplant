import argparse
import json
from pathlib import Path
import pickle
from collections import Counter, defaultdict
from hashlib import md5
import multiprocessing
#import multiprocessing.dummy as multiprocessing  # Use threads instead of processes to avoid pickling issues, since the function is not CPU-bound

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class NeighbourHoodDataset(Dataset):
    def __init__(self, neighbourhoods_dir: Path):
        self.neighbourhood_files = sorted(neighbourhoods_dir.glob("*.pkl"))
        neighbourhood_hash = md5()
        for neighbourhood_file in tqdm(self.neighbourhood_files):
            neighbourhood_hash.update(str(neighbourhood_file).encode('utf-8'))
            
        self.neighbourhood_hash = neighbourhood_hash.hexdigest()
    
        super().__init__()  
    
    def __len__(self):
        return len(self.neighbourhood_files)

    def __getitem__(self, index):
        neighbourhood_file = self.neighbourhood_files[index]
        neighbourhood = get_arrays_for_file(neighbourhood_file)
        return neighbourhood


def no_tensor_collator(batch):
    return batch

EVAL_METRICS = ('roc_auc', 'precision', 'recall', 'f1')

def main():
    parser = argparse.ArgumentParser(description="Analyze the neighbourhoods extracted from a vector database for a given test dataset.")
    parser.add_argument('neighbourhoods', help='The directory containing the neighbourhoods extracted from the vector database. This should be the output directory of the `extract_neighbourhoods` script.', type=Path)
    parser.add_argument('--output-dir', help='The directory to write the analysis results to.', type=Path)
    parser.add_argument('--num-workers', help='The number of workers to use for processing the neighbourhoods.', type=int, default=0)
    parser.add_argument('--recalculate', help='If set, recalculate the votes file HDF5 store', action='store_true')
    parser.add_argument('--threshold-metric', help="What metric to use for setting the threshold", choices=('f1', 'ba'), default='f1')
    parser.add_argument('--chunk-size', help="How large chunks of data to write to store", type=int, default=2**16)
    args = parser.parse_args()
    
    
    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    #  We'll start by extracting all the statistics for the neighbourhoods into ndarrays. We'll work under the assumption that they will fit in memory´
    
    neighbours_dataset = NeighbourHoodDataset(args.neighbourhoods)
    votes_file = output_dir / f"neigbourhood_analysis_{neighbours_dataset.neighbourhood_hash}.h5"
    if args.recalculate or not votes_file.exists():
        partial_file: Path = votes_file.with_suffix('.tmp')
        dataloader = DataLoader(neighbours_dataset, batch_size=1, num_workers=args.num_workers, drop_last=False, collate_fn=no_tensor_collator)
        with h5py.File(partial_file, 'w') as store:
            n_words = 0
            query_word_classes = []
            votes = []
            for batch in tqdm(dataloader):
                for example in batch:
                    batch_query_word_classes = example['query_word_classes']
                    n_batch_words = len(batch_query_word_classes)
                    n_words += n_batch_words
                    query_word_classes.append(batch_query_word_classes)
                    batch_votes = example['votes']
                    votes.append(batch_votes)
                    if n_words >= args.chunk_size:
                        n_words, query_word_classes, votes = record_results(store, query_word_classes, votes, args.chunk_size)
            
            # We'll do a sweep on the number of neighbours and the cosine similarity threshold to 
            # analyze the sensitivity and specificity of the neighbourhoods.
            #roc_auc_scores = compute_nearest_neighbour_rocauc(query_word_classes, neighbourhood_classes, n_neighbours)
            record_results(store, query_word_classes, votes, args.chunk_size)
        partial_file.rename(votes_file)

    metrics_file = output_dir / "analyzed_neighbourhoods.csv"
    if not metrics_file.exists() or args.recalculate:
        metrics_df = compute_statistics(votes_file, num_workers=args.num_workers)
        metrics_df.to_csv(metrics_file, index=False)
    else:
        metrics_df = pd.read_csv(metrics_file)

    for eval_metric in EVAL_METRICS:
        for threshold_on, threshold_df in metrics_df.groupby('threshold_on'):
            plt.figure()
            for weight_type, scores_df in threshold_df.groupby('weight_type'):
                scores_df = scores_df.sort_values('n_neighbours')
                n_neighbours = scores_df['n_neighbours']
                score = scores_df[eval_metric]
                plt.plot(n_neighbours, score, label=weight_type)
            plt.xlabel("Number of Neighbours")
            plt.ylabel(f"{eval_metric} Score")
            plt.title(f"{eval_metric} vs Number of Neighbours (thresholded on {threshold_on})")
            plt.legend()
            
            plt.savefig(output_dir / f"{threshold_on}_{eval_metric}_vs_neighbours.png")
            plt.show()

def record_results(store: h5py.File, query_word_classes, votes, chunk_size):
    # First we prepare the data by concatenating all arrays
    concatenated_query_word_classes = np.concatenate(query_word_classes)
    store_query_word_classes = concatenated_query_word_classes[:chunk_size]
    remaining_query_word_classes_arr = concatenated_query_word_classes[chunk_size:]
    n_remaining = len(remaining_query_word_classes_arr)
    remaining_word_classes = []
    if n_remaining > 0:
        remaining_word_classes.append(remaining_query_word_classes_arr)

    if 'query_word_classes' not in store:
        ds = store.create_dataset('query_word_classes', data=store_query_word_classes, maxshape=(None,), chunks=(chunk_size,))
    else:
        ds = store['query_word_classes']
        current_size = ds.shape[0]
        new_size = current_size + len(store_query_word_classes)
        ds.resize(new_size, axis=0)
        ds[current_size:new_size] = store_query_word_classes
    #max_neighbours = max(max_neighbours, partial_results["max_neighbours"])
    
    # The votes is a list of dictionaries. Each dictionary is from weighting_function 
    # to neighbourhood_votes. The neighbourhood votes is a dictionary with one key per 
    # neighbourhood size to the array of votes for that size.
    # We start by flattening the batched votes
    flattened_votes = defaultdict(lambda: defaultdict(list))
    for votes_batch in votes:
        for weighting_function, neighbour_votes in votes_batch.items():
            for n, votes_array in  neighbour_votes.items():
                flattened_votes[weighting_function][n].append(votes_array)
    
    remaining_votes_batch = {}
    any_remaining_votes = False
    for weighting_function, neighbour_votes in flattened_votes.items():
        g = store.require_group(weighting_function)
        all_n_neighbours = set(int(n) for n in neighbour_votes.keys())
        if 'n_neighbours' in g.attrs:
            all_n_neighbours.update(g.attrs['n_neighbours'])
        g.attrs['n_neighbours'] = sorted(all_n_neighbours)
        
        remaining_neighbour_votes = {}

        for n_neighbours, votes_value in neighbour_votes.items():
            concatenated_vote_values = np.concatenate(votes_value)
            store_vote_values = concatenated_vote_values[:chunk_size]
            remaining_vote_values_arr = concatenated_vote_values[chunk_size:]

            if  str(n_neighbours) not in g:
                g.create_dataset(str(n_neighbours), data=store_vote_values, maxshape=(None,), chunks=(chunk_size,))
            else:
                ds = g[str(n_neighbours)]
                current_size = ds.shape[0]
                new_size = current_size + store_vote_values.shape[0]
                ds.resize(new_size, axis=0)
                ds[current_size:new_size] = store_vote_values
            n_remaining_votes = len(remaining_vote_values_arr)
            if n_remaining_votes != n_remaining:
                raise RuntimeError("The remaining number of votes and number of remaining word classes differ")
            if n_remaining_votes > 0:
                remaining_neighbour_votes[n_neighbours] = remaining_vote_values_arr
                any_remaining_votes = True
        remaining_votes_batch[weighting_function] = remaining_neighbour_votes
    
    remaining_votes = []
    if any_remaining_votes:
        remaining_votes.append(remaining_votes_batch)

    weighting_functions = set(flattened_votes.keys())
    if 'weighting_functions' in store.attrs:
        weighting_functions.update(store.attrs['weighting_function'])
    store.attrs['weighting_function'] = sorted(weighting_functions)

    return n_remaining, remaining_word_classes, remaining_votes

def compute_statistics(votes_file: Path, num_workers=0):
    work_packages = []
    with h5py.File(votes_file, 'r') as store:
        weighting_functions = store.attrs['weighting_function']
        
        for weight_type in weighting_functions:
            g = store[weight_type]
            all_n_neighbours = g.attrs['n_neighbours']
            for n_neighbours in all_n_neighbours:
                work_package = (votes_file, weight_type, n_neighbours)
                work_packages.append(work_package)
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            records = list(tqdm(pool.imap_unordered(statistics_worker, work_packages), total=len(work_packages)))
    else:
        records = [statistics_worker(work_package) for work_package in tqdm(work_packages)]
    # The statistics worker returns a list of records, so we flatten this nested structure
    df = pd.DataFrame.from_records([record for record_pair in records for record in record_pair])
    return df

def statistics_worker(work_package):
    store_file, weight_type, n_neighbours = work_package
    records = []
    with h5py.File(store_file) as store:
        query_word_classes = store['query_word_classes'][:]
        votes = store[weight_type][str(n_neighbours)][:]
        fpr, tpr, roc_thresholds = roc_curve(query_word_classes, votes)
        roc_auc = np.trapezoid(tpr, fpr)
        precision_sweep, recall_sweep, prc_thresholds = precision_recall_curve(query_word_classes, votes)
        ap = -np.sum(np.diff(recall_sweep) * precision_sweep[:-1])

        # Figure out which threshold gives us the best Youden's J statistic (sensitivity + specificity - 1)
        # Since fpr is 1 - specificity and tpr is sensitivity, Youden's J can be calculated as tpr - fpr
        youdens_j = tpr - fpr
        best_threshold_index = np.argmax(youdens_j)
        best_threshold = roc_thresholds[best_threshold_index]
        discretized_votes = votes > best_threshold
        #precision, recall, f1_score, support = precision_recall_fscore_support(query_word_classes, discretized_votes, beta=1)
        precision = precision_score(query_word_classes, discretized_votes)
        f1 = f1_score(query_word_classes, discretized_votes)
        recall = recall_score(query_word_classes, discretized_votes)
        performance_record_ba = {'weight_type': weight_type,
                                'n_neighbours': n_neighbours,
                                'roc_auc': roc_auc,
                                'average_precision': ap, 
                                'threshold': best_threshold, 
                                'precision': precision, 
                                'recall': recall, 
                                'f1': f1,
                                'threshold_on': 'ba'}
        records.append(performance_record_ba)
        
        
        
        # precision/recall are length = len(thresholds)+1
        f1_sweep = 2 * precision_sweep[:-1] * recall_sweep[:-1] / (precision_sweep[:-1] + recall_sweep[:-1] + 1e-12)
        best_threshold_index = np.argmax(f1_sweep)
        best_threshold = prc_thresholds[best_threshold_index]
        discretized_votes = votes > best_threshold
        #precision, recall, f1_score, support = precision_recall_fscore_support(query_word_classes, discretized_votes, beta=1)
        precision = precision_sweep[best_threshold_index]
        recall = recall_sweep[best_threshold_index]
        f1 = f1_sweep[best_threshold_index]
        performance_record_ba = {'weight_type': weight_type,
                                'n_neighbours': n_neighbours,
                                'roc_auc': roc_auc, 
                                'average_precision': ap, 
                                'threshold': best_threshold, 
                                'precision': precision, 
                                'recall': recall, 
                                'f1': f1,
                                'threshold_on': 'f1'}
        
        records.append(performance_record_ba)
    return records

def get_arrays_for_file(neighbourhood_file):
    neighbourhood_classes = []
    neighbourhood_distances = []
    query_word_classes = []
    max_neighbours = 0
    with open(neighbourhood_file, 'rb') as fp:
        neighbourhood_data = pickle.load(fp)
        neighbourhoods = neighbourhood_data["neighbourhoods"]
        class_mappings = neighbourhood_data["class_mapping"]
        #The pickled files contain a list of tuples in the form of 
        # (('query_word', label), [(cossim1, 'neighbour_word1', label1), (cossim2 'neighbour_word2', label2), ...])
        # We'll analyze the sensitivity and specificity for different choices of number of 
        # neighbours and different thresholds on the cosine similarity. We will also 
        # analyze the distribution of the cosine similarities for 
        # the relevant and non-relevant neighbours.

    skip_lables = (class_mappings['stop_word'], class_mappings['known_positive'])  # We won't include stop words in the analysis, since they are not relevant for the classification task and they have very different neighbourhoods
    n_query_words = len(neighbourhoods)
    for (query_word, query_label), neighbours in neighbourhoods:
        if query_label in skip_lables:
            continue

        max_neighbours = max(max_neighbours, len(neighbours))
        query_neighbour_classes = []
        query_distances = []
        for cossim, neighbour_word, neighbour_label in neighbours:
            query_neighbour_classes.append(neighbour_label)
            query_distances.append(cossim)
        if query_neighbour_classes: # Stop words will have empty lists, we shouldn't add them
            query_word_classes.append(query_label)
            neighbourhood_classes.append(query_neighbour_classes)
            neighbourhood_distances.append(query_distances)
    
    neighbourhood_classes = np.array(neighbourhood_classes, dtype=np.int8)
    neighbourhood_distances = np.array(neighbourhood_distances, dtype=np.float32)
    query_word_classes = np.array(query_word_classes, dtype=np.int8)
    if (len(query_word_classes) != len(neighbourhood_classes) 
        or len(neighbourhood_distances) != len(query_word_classes)
        or len(neighbourhood_distances) != len(neighbourhood_classes)):
        raise RuntimeError("Array lengths differs")

    n_neighbours = list(range(1, max_neighbours+1))  # We'll use these as indexing ranges, if we don't add by 1 we'll not include the max number of neighbours
    votes = get_votes(neighbourhood_classes, neighbourhood_distances, n_neighbours)

    return {"query_word_classes": query_word_classes,
            "votes": votes}

def get_votes(neighbourhood_classes, neighbourhood_distances, n_neighbours):
    votes = {}
    for weight_type in ['constant', 'inverse', 'exponential']:
        weight_type_votes = {}
        neighbour_weights = weighting_function(neighbourhood_distances, weight_type=weight_type)  # This should give us an array of the same shape as neighbourhood_distances with the weights for each neighbour
        neighbour_weighted_classes = neighbour_weights*neighbourhood_classes
        # for n in n_neighbours:
        #     closest_neighbours = neighbour_weights[:, :n]
        #     weighted_neighbour_classes = closest_neighbours * neighbourhood_classes[:, :n]
        #     summed_class = np.sum(weighted_neighbour_classes, axis=1)
        #     votes[(weight_type, n)] = summed_class
        
        neighbour_cumsums = np.cumsum(neighbour_weighted_classes, axis=1)  # This gives as an array of the same shape as neighbour_weights where, for each row, each column contain the sum up to that column
        for i in n_neighbours:
            # Since one neigbhour corresonds to the first (zeroth) column, 
            # we need to subtract 1 to map the columns correctly
            weight_type_votes[i] = neighbour_cumsums[:, i-1]
        votes[weight_type] = weight_type_votes

    return votes


def weighting_function(distances, weight_type='inverse', eps=1e-5):
    if weight_type == 'inverse':
        return 1 / (distances + eps)  # Adding a small value to avoid division by zero
    elif weight_type == 'exponential':
        return np.exp(-distances)
    elif weight_type == 'constant':
        #n_neighbours = distances.shape[1]
        # This will give equal weight to all neighbours, while only equalling the 
        # mean in when using all neighbours. If we use a subset of the neighbours, this will give us a weighted mean where the weights sum to 1.
        return np.ones_like(distances)
        #return np.full_like(distances, 1.0 / n_neighbours)  
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    



if __name__ == "__main__":
    main()