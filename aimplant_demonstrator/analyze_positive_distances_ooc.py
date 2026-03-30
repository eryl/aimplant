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
import seaborn as sns

class NeighbourHoodDataset(Dataset):
    def __init__(self, neighbourhood_files, neighbourhood_limit=None):
        self.neighbourhood_files = sorted(neighbourhood_files)
        self.neighbourhood_limit = neighbourhood_limit
        super().__init__()  
    
    def __len__(self):
        return len(self.neighbourhood_files)

    def __getitem__(self, index):
        neighbourhood_file = self.neighbourhood_files[index]
        neighbourhood = get_arrays_for_file(neighbourhood_file, neighbourhood_limit=self.neighbourhood_limit)
        return neighbourhood


def no_tensor_collator(batch):
    return batch

EVAL_METRICS = ('roc_auc', 'precision', 'recall', 'f1', 'average_precision')

def main():
    parser = argparse.ArgumentParser(description="Analyze the neighbourhoods extracted from a vector database for a given test dataset.")
    parser.add_argument('neighbourhoods', help='The directory containing the neighbourhoods extracted from the vector database. This should be the output directory of the `extract_neighbourhoods` script.', type=Path)
    parser.add_argument('--output-dir', help='The directory to write the analysis results to.', type=Path)
    parser.add_argument('--num-workers', help='The number of workers to use for processing the neighbourhoods.', type=int, default=0)
    parser.add_argument('--recalculate', help='If set, recalculate the votes file HDF5 store', action='store_true')
    parser.add_argument('--threshold-metric', help="What metric to use for setting the threshold", choices=('f1', 'ba'), default='f1')
    parser.add_argument('--chunk-size', help="How large chunks of data to write to store", type=int, default=2**16)
    args = parser.parse_args()
    
    neighbourhood_files = sorted(args.neighbourhoods.glob("*.pkl"))
    neighbourhood_hash = md5()
    for neighbourhood_file in tqdm(neighbourhood_files):
        neighbourhood_hash.update(str(neighbourhood_file).encode('utf-8'))
        
    neighbourhood_hash = neighbourhood_hash.hexdigest()
    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    #  We'll start by extracting all the statistics for the neighbourhoods into ndarrays. We'll work under the assumption that they will fit in memory´
    
    
    output_dir.mkdir(exist_ok=True, parents=True)
    votes_file = output_dir / f"neigbourhood_distances_{neighbourhood_hash}.h5"
    if args.recalculate or not votes_file.exists():
        partial_file: Path = votes_file.with_suffix('.tmp')
        neighbourhood_limit = get_neighbourhood_limit(neighbourhood_files, num_workers=args.num_workers)
        if neighbourhood_limit is None or neighbourhood_limit < 0:
            raise RuntimeError(f"Could not determine the neighbourhood limit, got {neighbourhood_limit}. This means that all neighbourhoods are empty, which is unexpected.")

        neighbours_dataset = NeighbourHoodDataset(neighbourhood_files, neighbourhood_limit=neighbourhood_limit)
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
                    batch_distances = example['distances']
                    votes.append(batch_distances)
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
            plt.title(f"{eval_metric} vs Number of Neighbours (thresholded on {threshold_on})\nPositive examples: {threshold_df['positive_words'].iloc[0]}, Negative examples: {threshold_df['negative_words'].iloc[0]}")
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
        for centrality_measures, neighbour_votes in votes_batch.items():
            for n, votes_array in  neighbour_votes.items():
                flattened_votes[centrality_measures][n].append(votes_array)
    
    remaining_votes_batch = {}
    any_remaining_votes = False
    for centrality_measures, neighbour_votes in flattened_votes.items():
        g = store.require_group(centrality_measures)
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
        remaining_votes_batch[centrality_measures] = remaining_neighbour_votes
    
    remaining_votes = []
    if any_remaining_votes:
        remaining_votes.append(remaining_votes_batch)

    centrality_measures = set(flattened_votes.keys())
    if 'centrality_measures' in store.attrs:
        centrality_measures.update(store.attrs['centrality_measures'])
    store.attrs['centrality_measures'] = sorted(centrality_measures)

    return n_remaining, remaining_word_classes, remaining_votes

def record_results_flat(store: h5py.File, query_word_classes, votes, chunk_size):
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
    flattened_distances = defaultdict(list)
    for distances_batch in votes:
        for centrality_measure, distances in distances_batch.items():
            flattened_distances[centrality_measure].append(distances)
    
    remaining_distances_batch = {}
    any_remaining_distances = False
    for centrality_measure, neighbour_distances in flattened_distances.items():
        concatenated_distances = np.concatenate(neighbour_distances)
        store_distances = concatenated_distances[:chunk_size]
        remaining_distances = concatenated_distances[chunk_size:]

        if  centrality_measure not in store:
            store.create_dataset(centrality_measure, data=store_distances, maxshape=(None,), chunks=(chunk_size,))
        else:
            ds = store[centrality_measure]
            current_size = ds.shape[0]
            new_size = current_size + store_distances.shape[0]
            ds.resize(new_size, axis=0)
            ds[current_size:new_size] = store_distances
        n_remaining_distances = len(remaining_distances)
        if n_remaining_distances != n_remaining:
            raise RuntimeError("The remaining distances and number of remaining word classes differ")
        if n_remaining_distances > 0:
            any_remaining_distances = True
        remaining_distances_batch[centrality_measure] = remaining_distances
    
    remaining_distances = []
    if any_remaining_distances:
        remaining_distances.append(remaining_distances_batch)

    centrality_measure = set(flattened_distances.keys())
    if 'centrality_measures' in store.attrs:
        centrality_measure.update(store.attrs['centrality_measures'])
    store.attrs['centrality_measures'] = sorted(centrality_measure)

    return n_remaining, remaining_word_classes, remaining_distances


def compute_statistics_flat(votes_file: Path, num_workers=0):
    work_packages = []
    with h5py.File(votes_file, 'r') as store:
        centrality_measures = store.attrs['centrality_measures']
        
        for centrality_measure in centrality_measures:
            work_package = (votes_file, centrality_measure)
            work_packages.append(work_package)
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            records = list(tqdm(pool.imap_unordered(statistics_worker, work_packages), total=len(work_packages)))
    else:
        records = [statistics_worker(work_package) for work_package in tqdm(work_packages)]
    # The statistics worker returns a list of records, so we flatten this nested structure
    df = pd.DataFrame.from_records([record for record_pair in records for record in record_pair])
    return df



def compute_statistics(votes_file: Path, num_workers=0):
    work_packages = []
    with h5py.File(votes_file, 'r') as store:
        centrality_measures = store.attrs['centrality_measures']
        
        for weight_type in centrality_measures:
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
        # The sklearn metrics expects the scoring function to be increasing with the probability of being 
        # a positive example, so if we are using distances we need to invert them to get the votes. We 
        # also need to make sure that the votes are in the range [0, 1], so we can use them as probabilities. 
        # We can do this by transforming them with the exponentiated negative of the distance, which will 
        # give us values in the range (0, 1] that decrease with increasing distance.
        query_word_classes = store['query_word_classes'][:]
        positive_words = np.sum(store['query_word_classes'][:])
        negative_words = len(query_word_classes) - positive_words
        votes = np.exp(-store[weight_type][str(n_neighbours)][:])
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
                                'positive_words': positive_words,
                                'negative_words': negative_words,
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
                                'positive_words': positive_words,
                                'negative_words': negative_words,
                                'roc_auc': roc_auc, 
                                'average_precision': ap, 
                                'threshold': best_threshold, 
                                'precision': precision, 
                                'recall': recall, 
                                'f1': f1,
                                'threshold_on': 'f1'}
        
        records.append(performance_record_ba)
    return records

def get_arrays_for_file(neighbourhood_file, neighbourhood_limit=-1):
    neighbourhood_classes = []
    neighbourhood_distances = []
    query_word_classes = []
    
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
        
        query_neighbour_classes = []
        query_distances = []
        for cossim, neighbour_word, neighbour_label in neighbours:
            query_neighbour_classes.append(neighbour_label)
            query_distances.append(cossim)
        if query_neighbour_classes: # Stop words will have empty lists, we shouldn't add them
            query_word_classes.append(query_label)
            neighbourhood_classes.append(query_neighbour_classes)
            neighbourhood_distances.append(query_distances)
    
    # We slice the neighbourhoods to the max number of neighbours, so that we can stack them into ndarrays.
    try:
        neighbourhood_classes = np.array([neighbourhood_class[:neighbourhood_limit] for neighbourhood_class in neighbourhood_classes] , dtype=np.int8)
        neighbourhood_distances = np.array([neighbourhood_distance[:neighbourhood_limit] for neighbourhood_distance in neighbourhood_distances], dtype=np.float32)
    except ValueError as e:
        print(f"Error converting neighbourhood classes or distances to numpy arrays: {e}")
        print(f"Neighbourhood classes: {neighbourhood_classes}")
        print(f"Neighbourhood distances: {neighbourhood_distances}")
        print(f"File: {neighbourhood_file}")
        raise e
    query_word_classes = np.array(query_word_classes, dtype=np.int8)
    if (len(query_word_classes) != len(neighbourhood_classes) 
        or len(neighbourhood_distances) != len(query_word_classes)
        or len(neighbourhood_distances) != len(neighbourhood_classes)):
        raise RuntimeError("Array lengths differs")

    n_neighbours = list(range(1, neighbourhood_limit+1))  # We'll use these as indexing ranges, if we don't add by 1 we'll not include the max number of neighbours
    distances = get_central_distance(neighbourhood_classes, neighbourhood_distances, n_neighbours)

    return {"query_word_classes": query_word_classes,
            "distances": distances}

def get_central_distance(neighbourhood_classes, neighbourhood_distances, n_neighbours):
    distances = {}
    for measure, measure_fun in [('mean', np.mean), ('median', np.median)]:
        distance_per_neighbourhood = {}
        
        for i in n_neighbours:
            distance_per_neighbourhood[i] = measure_fun(neighbourhood_distances[:, :i], axis=1)
        distances[measure] = distance_per_neighbourhood

    return distances


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
    

def get_min_neighbours_for_file(neighbourhood_file):
    """
    Get the minimum number of neighbours for a given neighbourhood file (greater than 0). This is used for determining the maximum number of neighbours 
    to analyze across all files, since some files might have very few neighbours for some query words, and we want to include all query words in the analysis.
    """
    min_neighbours = None
    with open(neighbourhood_file, 'rb') as fp:
        neighbourhood_data = pickle.load(fp)
        neighbourhoods = neighbourhood_data["neighbourhoods"]
        for (query_word, query_label), neighbours in neighbourhoods:
            # Skip words will have an empty neighbourhood, since they won't contribute to 
            # the analysis and they would cause issues with stacking the arrays into ndarrays.
            n_neighbours_for_word = len(neighbours)
            if n_neighbours_for_word > 0 and (min_neighbours is None or n_neighbours_for_word < min_neighbours):
                min_neighbours = n_neighbours_for_word
    return min_neighbours

def get_neighbourhood_limit(neighbourhoods_files, num_workers=0):
    min_neighbours = None
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for file_min_neighbours in tqdm(pool.imap_unordered(get_min_neighbours_for_file, neighbourhoods_files), total=len(neighbourhoods_files)):
                if file_min_neighbours is not None and (min_neighbours is None or file_min_neighbours < min_neighbours):
                    min_neighbours = file_min_neighbours
    else:
        for file_min_neighbours in tqdm(map(get_min_neighbours_for_file, neighbourhoods_files), total=len(neighbourhoods_files)):
            if file_min_neighbours is not None and (min_neighbours is None or file_min_neighbours < min_neighbours):
                min_neighbours = file_min_neighbours
    return min_neighbours

if __name__ == "__main__":
    main()