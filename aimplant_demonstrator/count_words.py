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
    
<<<<<<< HEAD
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
    common_count = 0
    for group1 in tqdm(groups1, desc="Finding common words"):
        for group2 in groups2:
            if jaro_winkler_similarity(group1[0], group2[0]) >= threshold:
=======
    # Use unique terms to avoid duplicate work.
    unique_list1 = list(dict.fromkeys(list1))

    # Keep unique terms from list2; grouping huge vocabularies is prohibitively slow.
    unique_list2 = list(dict.fromkeys(list2))

    # Simple index to reduce expensive similarity computations.
    by_initial = {}
    for word in unique_list2:
        if not word:
            continue
        by_initial.setdefault(word[0], []).append(word)
    
    # Find common grouped words
    common_count = 0
    common_words = []
    for ref_word in unique_list1:
        candidates = by_initial.get(ref_word[0], []) if ref_word else []

        for candidate in candidates:
            # Length filtering eliminates many impossible matches cheaply.
            if abs(len(ref_word) - len(candidate)) > 4:
                continue

            if jaro_winkler_similarity(ref_word, candidate) >= threshold:
>>>>>>> 2be9f92 (update)
                common_count += 1
                common_words.append(ref_word)
                break
    
    return {
        "list1_original_count": len(list1),
        "list1_grouped_count": len(unique_list1),
        "list2_original_count": len(list2),
        "list2_grouped_count": len(unique_list2),
        "common_words": common_words,
        "common_count": common_count
    }

# Example usage
if __name__ == "__main__":
<<<<<<< HEAD

    #glossary = ["color", "colour", "coler", "gray", "grey", "red"]
    #word_freqs = ["colour", "color", "gray", "blue", "red", "reed"]
    #glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
    #word_freqs = '/home/abragam23/fedhealth_data/word_frequencies.txt'
    #stop_words_file = '/home/abragam23/fedhealth_data/manual_stop_list.txt'
    glossary = 'glossary.txt'
    word_freqs = 'word_freq.txt'
    stop_words_file = 'stop_words.txt'
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f if line.strip()]        
        stop_words = set(stop_words + list(stopwords.words('swedish')))
        
=======
    glossary = 'glossary.txt'  # Path to glossary file
    word_freqs = 'word_freq.txt'  # Path to word frequencies file
    #glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
    #word_freqs = '/home/abragam23/fedhealth_data/word_frequencies.txt'
>>>>>>> 2be9f92 (update)
    # Read words from glossary.txt
    with open(glossary, 'r', encoding='utf-8') as f:
        words_a = [line.strip() for line in f if line.strip()]
        words_a = [word for word in words_a if word.lower() not in stop_words]
    
    # Read word_freq.txt and extract words with their frequencies
<<<<<<< HEAD
    words_b = {}
=======
    # Supports CSV files like: term,count
    words_b = []
>>>>>>> 2be9f92 (update)
    with open(word_freqs, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        f.seek(0)

        # Parse as CSV when commas are present in the sample, otherwise use whitespace splitting.
        if ',' in sample:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                word = row[0].strip()
                if not word or word.lower() == 'term':
                    continue

                # Keep one entry per term for overlap/similarity checks.
                words_b.append(word)
        else:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                word = parts[0]
<<<<<<< HEAD
                freq = int(parts[1]) if len(parts) > 1 else 1
                words_b[word] = freq
    #words_b = {word: freq for word, freq in words_b.items() if word.lower() not in stop_words}
=======
                # Keep one entry per term for overlap/similarity checks.
                words_b.append(word)
>>>>>>> 2be9f92 (update)

    result = compare_word_lists(words_a, list(words_b.keys()))
    print(f"Glossary - Original: {result['list1_original_count']}, Grouped: {result['list1_grouped_count']}")
<<<<<<< HEAD
    print(f"Word Frequency List - Unique: {result['list2_original_count']}, Grouped: {result['list2_grouped_count']}, Total tokens: {sum(words_b.values())}")
    print(f"Common words: {result['common_words']}")
=======
    print(f"Word Frequency List - Original: {result['list2_original_count']}, Grouped: {result['list2_grouped_count']}")
    print(f"Common words: {result['common_words']}, common percentage: {result['common_count'] / result['list1_grouped_count'] * 100:.2f}%")
>>>>>>> 2be9f92 (update)
