import csv
from nltk.metrics.distance import jaro_winkler_similarity
from tqdm import tqdm

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
        "glossary_original": list1,
        "glossary_grouped": groups1,
        "freq_original": list2,
        "freq_grouped": groups2,
        "common_words": common_words,
    }

def summarize_and_save(grouped_list, output_file):
    summarized = {}
    for word in grouped_list:
        rep_word = word[0]  # Representative word for the group
        total_freq = len(word)  # Count of words in the group
        summarized[rep_word] = total_freq
    with open(output_file, 'w', encoding='utf-8') as f:
        for rep_word, total_freq in sorted(summarized.items(), key=lambda x: x[1], reverse=True):
            group_words = next(g for g in grouped_list if g[0] == rep_word)
            if len(group_words) > 1:
                f.write(f"  {rep_word}: {total_freq}; {group_words}\n")
            else:
                f.write(f"  {rep_word}: {total_freq}\n")

if __name__ == "__main__":
    glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
    word_freqs = '/home/abragam23/fedhealth_data/word_frequencies.txt'
    #stop_words_file = '/home/abragam23/fedhealth_data/manual_stop_list.txt'
    #glossary = 'glossary.txt'
    #word_freqs = 'word_freq.txt'
    stop_words_file = 'combined_stop_words.txt'

    # Load stop words from file and combine with NLTK's Swedish stop words
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f if line.strip()]        
  
    # Read words from glossary.txt
    with open(glossary, 'r', encoding='utf-8') as f:
        glossary_words= [line.strip() for line in f if line.strip()]
        glossary_words = [word for word in glossary_words if word.lower() not in stop_words]
    
    # Read word_freq.txt and extract words with their frequencies
    freq_words = {}
    with open(word_freqs, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts or parts[0] == 'term':
                continue
            if parts[0] not in stop_words:
                word = parts[0]
                freq = int(parts[1]) 
                freq_words[word] = freq

    result = compare_word_lists(glossary_words, list(freq_words.keys()))

    print(f"Glossary - Original: {len(result['glossary_original'])}, Grouped: {len(result['glossary_grouped'])}")
    summarize_and_save(result['glossary_grouped'],'glossary_summary.txt')
    print(f"Word Frequency List - Unique: {len(result['freq_original'])}, Grouped: {len(result['freq_grouped'])}, Total tokens: {sum(freq_words.values())}")
    summarize_and_save(result['freq_grouped'],'word_freq_summary.txt')
    print(f"Common words: {len(result['common_words'])}:  {result['common_words']}")


