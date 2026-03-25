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
    #glossary = ["color", "colour", "coler", "gray", "grey", "red"]
    #word_freqs = ["colour", "color", "gray", "blue", "red", "reed"]
    glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
    word_freqs = '/home/abragam23/fedhealth_data/word_frequencies.txt'
    #glossary = 'glossary.txt'
    #word_freqs = 'word_freq.txt'
    # Read words from glossary.txt
    with open(glossary, 'r', encoding='utf-8') as f:
        words_a = [line.strip() for line in f if line.strip()]
    
    # Read word_freq.txt and extract words with their frequencies
    words_b = {}
    with open(word_freqs, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                word = parts[0]
                freq = int(parts[1]) if len(parts) > 1 else 1
                words_b[word] = freq

    result = compare_word_lists(words_a, list(words_b.keys()))
    print(f"Glossary - Original: {result['list1_original_count']}, Grouped: {result['list1_grouped_count']}")
    print(f"Word Frequency List - Unique: {result['list2_original_count']}, Grouped: {result['list2_grouped_count']}, Total tokens: {sum(words_b.values())}")
    print(f"Common words: {result['common_words']}")