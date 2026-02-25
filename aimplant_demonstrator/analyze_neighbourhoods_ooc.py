import argparse
import json
from pathlib import Path
import pickle
from collections import Counter, defaultdict
from hashlib import md5
import multiprocessing
#import multiprocessing.dummy as multiprocessing  # Use threads instead of processes to avoid pickling issues, since the function is not CPU-bound

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Analyze the neighbourhoods extracted from a vector database for a given test dataset.")
    parser.add_argument('neighbourhoods', help='The directory containing the neighbourhoods extracted from the vector database. This should be the output directory of the `extract_neighbourhoods` script.', type=Path)
    parser.add_argument('--output-dir', help='The directory to write the analysis results to.', type=Path)
    parser.add_argument('--num-workers', help='The number of workers to use for processing the neighbourhoods.', type=int, default=0)
    parser.add_argument('--skip-cache', help='Whether to use cached results if they exist. If not set, will always recompute the analysis.', action='store_true')
    parser.add_argument('--eval-metric', help="What metric to use for evaluation", choices=('roc_auc', 'threshold', 'precision', 'recall', 'f1'), default='f1')

    args = parser.parse_args()
    neighbourhood_files = sorted(args.neighbourhoods.glob("*.pkl"))

    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    #  We'll start by extracting all the statistics for the neighbourhoods into ndarrays. We'll work under the assumption that they will fit in memory´

    neighbourhood_hash = md5()
    for neighbourhood_file in neighbourhood_files:
        neighbourhood_hash.update(str(neighbourhood_file).encode('utf-8'))
    cache_files = f"neigbourhood_analysis_{neighbourhood_hash.hexdigest()}.pkl"
    if  (output_dir / cache_files).exists() and not args.skip_cache:
        with open(output_dir / cache_files, 'rb') as fp:
            cache_data = pickle.load(fp)
            query_votes = cache_data["votes"]
            query_word_classes = cache_data["query_word_classes"]
    else:
        query_word_classes = []
        max_neighbours = 0
        query_votes = defaultdict(lambda: defaultdict(list))

        if args.num_workers > 1:
            with multiprocessing.Pool(args.num_workers) as pool:
                for partial_results in tqdm(pool.imap_unordered(get_arrays_for_file, neighbourhood_files), total=len(neighbourhood_files), desc="Processing neighbourhood files"):
                    query_word_classes.append(partial_results["query_word_classes"])
                    #max_neighbours = max(max_neighbours, partial_results["max_neighbours"])
                    for weighting_function, neighbour_votes in partial_results["votes"].items():
                        for n_neighbours, votes_value in neighbour_votes.items():
                            query_votes[weighting_function][n_neighbours].append(votes_value)
        else:
            for partial_results in tqdm(map(get_arrays_for_file, neighbourhood_files), total=len(neighbourhood_files), desc="Processing neighbourhood files"):
                    query_word_classes.append(partial_results["query_word_classes"])
                    #max_neighbours = max(max_neighbours, partial_results["max_neighbours"])
                    for weighting_function, neighbour_votes in partial_results["votes"].items():
                        for n_neighbours, votes_value in neighbour_votes.items():
                            query_votes[weighting_function][n_neighbours].append(votes_value)

        #neighbourhood_classes = np.concatenate(neighbourhood_classes, axis=0)
        #neighbourhood_distances = np.concatenate(neighbourhood_distances, axis=0)
        query_word_classes = np.concatenate(query_word_classes, axis=0)
        query_votes = {weighting_function: {n_neighbours: np.concatenate(votes_list, axis=0) 
                                            for n_neighbours, votes_list in n_neighbours_dict.items()}
                                            for weighting_function, n_neighbours_dict in query_votes.items()}

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / cache_files, 'wb') as fp:
            pickle.dump({"votes": query_votes,
                         "query_word_classes": query_word_classes,
                         }, fp)

    # We'll do a sweep on the number of neighbours and the cosine similarity threshold to 
    # analyze the sensitivity and specificity of the neighbourhoods.
    #roc_auc_scores = compute_nearest_neighbour_rocauc(query_word_classes, neighbourhood_classes, n_neighbours)
    roc_auc_scores, best_score, best_hyper_params = compute_statistics(query_word_classes, query_votes, eval_metric=args.eval_metric)
    
    

    with open(output_dir / "analyzed_neighbourhoods.json", 'w') as fp:
        json.dump({"roc_auc_scores": roc_auc_scores,
                   "best_score": best_score,
                   "best_hyper_params": best_hyper_params}, fp, indent=2)

    plt.figure()
    for weight_type, scores_dict in roc_auc_scores.items():
        neighbours, scores_records = zip(*sorted(scores_dict.items()))
        scores = [record[args.eval_metric] for record in scores_records]
        plt.plot(neighbours, scores, label=weight_type)
    plt.xlabel("Number of Neighbours")
    plt.ylabel(f"{args.eval_metric} Score")
    plt.title(f"{args.eval_metric} vs Number of Neighbours")
    plt.legend()
    
    plt.savefig(output_dir / f"{args.eval_metric}_vs_neighbours.png")
    plt.show()



def compute_statistics(query_word_classes, query_votes, eval_metric='f1_score'):
    performance = {}
    best_score = float('-inf')
    best_hyper_params = None
    
    for weight_type, votes_dict in query_votes.items():
        performance[weight_type] = {}
        for n_neighbours, votes in votes_dict.items():
            # Compute ROC AUC score for this set of votes
            fpr, tpr, thresholds = roc_curve(query_word_classes, votes)
            roc_auc = np.trapezoid(tpr, fpr)
            # Figure out which threshold gives us the best Youden's J statistic (sensitivity + specificity - 1)
            # Since fpr is 1 - specificity and tpr is sensitivity, Youden's J can be calculated as tpr - fpr
            youdens_j = tpr - fpr
            best_threshold_index = np.argmax(youdens_j)
            best_threshold = thresholds[best_threshold_index]
            discretized_votes = votes > best_threshold
            #precision, recall, f1_score, support = precision_recall_fscore_support(query_word_classes, discretized_votes, beta=1)
            precision = precision_score(query_word_classes, discretized_votes)
            f1 = f1_score(query_word_classes, discretized_votes)
            recall = recall_score(query_word_classes, discretized_votes)
            performance_record = {'roc_auc': roc_auc, 
                                  'threshold': best_threshold, 
                                'precision': precision, 
                                'recall': recall, 
                                'f1': f1}
                                
            performance[weight_type][n_neighbours] = performance_record

            if performance_record[eval_metric] > best_score:
                best_score = roc_auc
                best_hyper_params = {"neighbours": n_neighbours, "weight_type": weight_type}
    return performance, best_score, best_hyper_params


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
    neighbourhood_distances = np.array(neighbourhood_distances)
    query_word_classes = np.array(query_word_classes, dtype=np.int8)

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