import csv

from nltk.metrics.distance import jaro_winkler_similarity
from tqdm import tqdm
from nltk.corpus import stopwords

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
        
        for word in tqdm(words, desc="Grouping words"):
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
    common_words = []
    for group1 in tqdm(groups1, desc="Finding common words"):
        for group2 in groups2:
            if jaro_winkler_similarity(group1[0], group2[0]) >= threshold:
                common_words.append(group1[0])
                break
    
    return {
        "list1_original": list1,
        "list1_grouped": groups1,
        "list2_original": list2,
        "list2_grouped": groups2,
        "common_words": common_words,
    }

# Example usage
if __name__ == "__main__":

    #glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
    #word_freqs = '/home/abragam23/fedhealth_data/word_frequencies.txt'
    #stop_words_file = '/home/abragam23/fedhealth_data/manual_stop_list.txt'
    glossary = 'glossary.txt'
    word_freqs = 'word_freq.txt'
    stop_words_file = 'stop_words.txt'
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f if line.strip()]        
        stop_words = set(stop_words + list(stopwords.words('swedish')))
        
    # Read words from glossary.txt
    with open(glossary, 'r', encoding='utf-8') as f:
        glossary_words= [line.strip() for line in f if line.strip()]
        glossary_words = [word for word in glossary_words if word.lower() not in stop_words]
    
    # Read word_freq.txt and extract words with their frequencies
    freq_words = {}
    with open(word_freqs, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            word = parts[0]
            freq = int(parts[1]) if len(parts) > 1 else 1
            freq_words[word] = freq
    #freq_words = {word: freq for word, freq in freq_words.items() if word.lower() not in stop_words}

    result = compare_word_lists(glossary_words, list(freq_words.keys()))
    print(f"Glossary - Original: {len(result['list1_original'])}, Grouped: {len(result['list1_grouped'])}")
    print(f"Word Frequency List - Unique: {len(result['list2_original'])}, Grouped: {len(result['list2_grouped'])}, Total tokens: {sum(freq_words.values())}")
    print(f"Common words: {len(result['common_words'])}:  {result['common_words']}")
