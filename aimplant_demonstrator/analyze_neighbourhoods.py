import argparse
import json
from pathlib import Path
import pickle
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Analyze the neighbourhoods extracted from a vector database for a given test dataset.")
    parser.add_argument('neighbourhoods', help='The directory containing the neighbourhoods extracted from the vector database. This should be the output directory of the `extract_neighbourhoods` script.', type=Path)
    parser.add_argument('--output-dir', help='The directory to write the analysis results to.', type=Path)
    args = parser.parse_args()
    neighbourhoods = args.neighbourhoods.glob("*.pkl")

    output_dir = args.output_dir if args.output_dir else args.neighbourhoods / "analysis"
    #  We'll start by extracting all the statistics for the neighbourhoods into ndarrays. We'll work under the assumption that they will fit in memory´
    neighbourhood_classes = []
    neighbourhood_distances = []
    query_word_classes = []
    max_neighbours = 0
    for neighbourhood_file in neighbourhoods:
        print(f"Analyzing {neighbourhood_file}...")
        with open(neighbourhood_file, 'rb') as fp:
            neighbourhood_data = pickle.load(fp)
        #The pickled files contain a list of tuples in the form of 
        # (('query_word', label), [(cossim1, 'neighbour_word1', label1), (cossim2 'neighbour_word2', label2), ...])
        # We'll analyze the sensitivity and specificity for different choices of number of 
        # neighbours and different thresholds on the cosine similarity. We will also 
        # analyze the distribution of the cosine similarities for 
        # the relevant and non-relevant neighbours.
        n_query_words = len(neighbourhood_data)
        for (query_word, query_label), neighbours in neighbourhood_data:
            query_label = 2*query_label - 1
            max_neighbours = max(max_neighbours, len(neighbours))
            query_neighbour_classes = []
            query_distances = []
            for cossim, neighbour_word, neighbour_label in neighbours:
                neighbour_label = 2*neighbour_label - 1 # We'll -1, 1 encode the labels
                query_neighbour_classes.append(neighbour_label)
                query_distances.append(cossim)
            if query_neighbour_classes: # Stop words will have empty lists, we shouldn't add them
                query_word_classes.append(query_label)
                neighbourhood_classes.append(query_neighbour_classes)
                neighbourhood_distances.append(query_distances)
    neighbourhood_classes = np.array(neighbourhood_classes, dtype=np.int8)
    neighbourhood_distances = np.array(neighbourhood_distances)
    query_word_classes = np.array(query_word_classes, dtype=np.int8)

    # We'll do a sweep on the number of neighbours and the cosine similarity threshold to 
    # analyze the sensitivity and specificity of the neighbourhoods.
    n_neighbours = list(range(1, max_neighbours+1))  # We'll use these as indexing ranges, if we don't add by 1 we'll not include the max number of neighbours
    #roc_auc_scores = compute_nearest_neighbour_rocauc(query_word_classes, neighbourhood_classes, n_neighbours)
    roc_auc_scores, best_score, best_hyper_params = compute_weighted_nearest_neighbour_rocauc(query_word_classes, 
                                                               neighbourhood_classes, 
                                                               neighbourhood_distances, 
                                                               n_neighbours, weighting_scheme=['inverse', 'exponential'])
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "analyzed_neighbourhoods.json", 'w') as fp:
        json.dump({"roc_auc_scores": roc_auc_scores,
                    "best_score": best_score,
                    "best_hyper_params": best_hyper_params}, fp, indent=2)

    plt.figure()
    for weight_type, scores_dict in roc_auc_scores.items():
        neighbours, scores = zip(*sorted(scores_dict.items()))
        roc_aucs, thresholds = zip(*scores)
        plt.plot(neighbours, roc_aucs, label=weight_type)
    plt.xlabel("Number of Neighbours")
    plt.ylabel("ROC AUC Score")
    plt.title("ROC AUC Score vs Number of Neighbours")
    plt.legend()
    
    plt.savefig(output_dir / "roc_auc_vs_neighbours.png")
    plt.show()

# def compute_nearest_neighbour_rocauc(query_word_classes, 
#                                      neighbourhood_classes, 
#                                      n_neighbours):
#     # The neighbourhood arrays are already sorted row-wise according to the distance.
#     # If the classes are encoded by -1 and 1, for negative and positive,
#     # adding the query class to the neighbour class would give 0 where they differ,
#     # -2 where they agree negative and 2 where they agree positive
#     best_score = float('-inf')
#     best_hyper_params = None
    
#     roc_scores_per_neighbour = {}
#     agreement = query_word_classes[:, np.newaxis] + neighbourhood_classes
#     for n in n_neighbours:
#         closest_neighbours = neighbourhood_classes[:, :n]
#         mean_class = np.mean(closest_neighbours, axis=1)
#         # We'll produce the ROC AUC score based on these mean classes
        
#         if roc_auc > best_score:
#             best_score = roc_auc
#             best_hyper_params = n
#     return roc_scores_per_neighbour




def compute_weighted_nearest_neighbour_rocauc(query_word_classes, 
                                              neighbourhood_classes, 
                                              neighbourhood_distances,
                                              n_neighbours, weighting_scheme=['inverse', 'exponential']):
    # The neighbourhood arrays are already sorted row-wise according to the distance.
    # If the classes are encoded by -1 and 1, for negative and positive,
    # adding the query class to the neighbour class would give 0 where they differ,
    # -2 where they agree negative and 2 where they agree positive
    roc_scores_per_neighbour = defaultdict(dict)
    best_score = float('-inf')
    best_hyper_params = None
    
    for weight_type in weighting_scheme:
        neighbour_weights = weighting_function(neighbourhood_distances, weight_type=weight_type)  # This should give us an array of the same shape as neighbourhood_distances with the weights for each neighbour
        for n in n_neighbours:
            closest_neighbours = neighbour_weights[:, :n]
            weighted_neighbour_classes = closest_neighbours * neighbourhood_classes[:, :n]
            mean_class = np.mean(weighted_neighbour_classes, axis=1)
            # We'll produce the ROC AUC score based on these mean classes
            fpr, tpr, thresholds = roc_curve(query_word_classes, mean_class)
            # We calculate the AUC using the trapezoidal rule
            roc_auc = np.trapezoid(tpr, fpr)
            # Figure out which threshold gives us the best Youden's J statistic (sensitivity + specificity - 1)
            # Since fpr is 1 - specificity and tpr is sensitivity, Youden's J can be calculated as tpr - fpr
            youdens_j = tpr - fpr
            best_threshold_index = np.argmax(youdens_j)
            best_threshold = thresholds[best_threshold_index]
            roc_scores_per_neighbour[weight_type][n] = (roc_auc, best_threshold)
            if roc_auc > best_score:
                best_score = roc_auc
                best_hyper_params = {"neighbours": n, "weight_type": weight_type}
    return roc_scores_per_neighbour, best_score, best_hyper_params

def weighting_function(distances, weight_type='inverse', eps=1e-5):
    if weight_type == 'inverse':
        return 1 / (distances + eps)  # Adding a small value to avoid division by zero
    elif weight_type == 'exponential':
        return np.exp(-distances)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    

def compute_metrics(neighbourhood_classes, neighbourhood_distances, query_word_classes, n_neighbours, threshold):
    # Compute sensitivity and specificity for a given number of neighbours and threshold
    # This is a simplified version that assumes all query words are relevant (i.e., we're computing metrics for a binary classification task)
    # In a more complex scenario, we would need to know which query words are actually relevant vs. non-relevant.
    # For now, we assume all query words are relevant.
    n_query_words = len(query_word_classes)
    n_relevant = np.sum(query_word_classes == 1)  # Assuming 1 means relevant
    n_non_relevant = n_query_words - n_relevant

    # Select top n_neighbours with cosine similarity above threshold
    selected_indices = np.where(neighbourhood_distances > threshold)[0]
    selected_neighbourhood_classes = neighbourhood_classes[selected_indices]

    # Count true positives, false positives, false negatives, true negatives
    tp = np.sum((selected_neighbourhood_classes == 1) & (query_word_classes == 1))
    fp = np.sum((selected_neighbourhood_classes == 1) & (query_word_classes == 0))
    fn = np.sum((selected_neighbourhood_classes == 0) & (query_word_classes == 1))
    tn = np.sum((selected_neighbourhood_classes == 0) & (query_word_classes == 0))

    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0

    return sensitivity, specificity


if __name__ == "__main__":
    main()