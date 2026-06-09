items = set()
glossary = '/home/abragam23/fedhealth_data/Glossary_only_known_implants.txt'
with open('word_freq_summary_085.txt', 'r', encoding='utf-8') as f:
    with open(glossary, 'r', encoding='utf-8') as g:
        glossary = [line.strip() for line in g if line.strip()]
        print(f"Loaded {len(glossary)} items from {glossary}")
        for line in f:
            item = line.split(':')[0].strip()
            if item in glossary:
                print(f"Item in glossary: {item}")
                items.add(line)

output_file = 'implants_in_word_freq.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in items:
        f.write(str(item) + '\n')

print(f"Wrote {len(items)} items to {output_file}")