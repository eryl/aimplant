from nltk.metrics import edit_distance
from nltk.metrics.distance import jaro_winkler_similarity

def compare_word_lists(list1, list2, threshold=0.85):
    """
    Compare two word lists using Jaro-Winkler similarity.
    
    Args:
        list1: First list of words
        list2: Second list of words
        threshold: Jaro-Winkler similarity threshold for grouping (0-1)
    
    Returns:
        Dictionary with comparison statistics
    """
    
    def group_similar_words(words, threshold):
        """Group similar words based on Jaro-Winkler similarity."""
        groups = []
        used = set()
        
        for word in words:
            if word in used:
                continue
            group = [word]
            used.add(word)
            
            for other_word in words:
                if other_word not in used:
                    similarity = jaro_winkler_similarity(word, other_word)
                    if similarity >= threshold:
                        group.append(other_word)
                        used.add(other_word)
            
            groups.append(group)
        
        return groups
    
    # Group words in both lists
    groups1 = group_similar_words(list1, threshold)
    groups2 = group_similar_words(list2, threshold)
    
    # Find common grouped words
    common_count = 0
    for group1 in groups1:
        for group2 in groups2:
            if jaro_winkler_similarity(group1[0], group2[0]) >= threshold:
                common_count += 1
                break
    
    return {
        "list1_original_count": len(list1),
        "list1_grouped_count": len(groups1),
        "list2_original_count": len(list2),
        "list2_grouped_count": len(groups2),
        "common_words": common_count
    }

# Example usage
if __name__ == "__main__":
    words_a = ["color", "colour", "coler", "gray", "grey", "red"]
    words_b = ["colour", "color", "gray", "blue", "red", "reed"]
    
    result = compare_word_lists(words_a, words_b)
    print(f"List 1 - Original: {result['list1_original_count']}, Grouped: {result['list1_grouped_count']}")
    print(f"List 2 - Original: {result['list2_original_count']}, Grouped: {result['list2_grouped_count']}")
    print(f"Common words: {result['common_words']}")