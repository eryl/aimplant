with open('word_freq_summary.txt', 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    list_part = first_line.split(';', 1)[1].strip()
    date_items = eval(list_part)

with open('combined_stop_words.txt', 'r', encoding='utf-8') as f:
    items = f.read().splitlines()
    items.extend(date_items)

output_file = 'combined_date_stopwords.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in items:
        f.write(str(item) + '\n')

print(f"Wrote {len(items)} items to {output_file}")
