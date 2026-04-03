import csv
from nltk.metrics.distance import binary_distance
from tqdm import tqdm

def compare_word_lists(list1, list2):
    ""
    # Find common grouped words
    common_words = []
    for word1 in tqdm(list1, desc="Finding common words"):
        for word2 in list2:
            if binary_distance(word1, word2) == 0:
                common_words.append(word1)
                break
    
    return common_words

if __name__ == "__main__":
    #glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
    #word_freqs = '/home/abragam23/fedhealth_data/word_frequencies.txt'
    #stop_words_file = '/home/abragam23/fedhealth_data/manual_stop_list.txt'
    glossary = 'glossary.txt'
    word_freqs = 'word_freq.txt'
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

    print(f"Glossary - Original: {len(glossary_words)}")
    print(f"Word Frequency List - Unique: {len(freq_words)}")
    print(f"Common words: {len(result)}:  {result}")


