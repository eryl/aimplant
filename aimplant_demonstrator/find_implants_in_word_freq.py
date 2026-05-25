items = set()
with open('word_freq_summary_092.txt', 'r', encoding='utf-8') as f:
    with open('glossary.txt', 'r', encoding='utf-8') as g:
        glossary = [line.strip() for line in g if line.strip()]
        print(f"Loaded {len(glossary)} items from glossary.txt")
        for line in f:
            item = line.split(':')[0].strip()
            print(f"Checking item: {item}")
            if item in glossary:
                items.add(line)

output_file = 'implants_in_word_freq.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in items:
        f.write(str(item))

print(f"Wrote {len(items)} items to {output_file}")