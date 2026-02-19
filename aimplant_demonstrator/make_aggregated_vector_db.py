import argparse
from collections import defaultdict
from pathlib import Path
import json
import csv
import os
from functools import partial
from typing import Optional
import multiprocessing
import pickle
import json
from collections import Counter
import math

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import trange, tqdm
from datasets import load_dataset
import h5py
import numpy as np
from sqlitedict import SqliteDict
import lancedb
import pyarrow as pa
#from lancedb.exceptions import LanceError

def collate_batch(batch, tokenizer):
    keys = batch[0].keys()
    collated = {k: [] for k in keys}
    for b in batch:
        for k in keys:
            collated[k].append(b[k])
    padded = tokenizer.pad({"input_ids": collated["input_ids"], "special_tokens_mask": collated["special_tokens_mask"]}, return_tensors="pt")
    collated.update(padded)
    return collated


def main():
    parser = argparse.ArgumentParser(description="Create new table in the database with the aggregated vectors from the existing table.")
    parser.add_argument('vector_database',
                        help=("Path to directory of the lancedb database. Apart from the directory for the "
                             "database, there should be a JSON file in the parent directory with the same base "
                             "name as the directory with the `.json` suffix"),
                        type=Path)
    parser.add_argument('--batch-size', help="Size of batches.", type=int, default=1024)
    parser.add_argument('--query_size', 
                        help="The data is collected in chunks before being written to disk. This decides the size of those chunks", 
                        type=int, 
                        default=128)
    args = parser.parse_args()
    
    database_metadata_path = args.vector_database.with_suffix('.json')
    with open(database_metadata_path) as fp:
        database_metadata = json.load(fp)
    aggregation = database_metadata['aggregation']
    layers = [int(l) for l in database_metadata['layers']]
    table_name = database_metadata.get('table_name', 'words')
    
    db = lancedb.connect(args.vector_database)
    table = db.open_table(table_name)

    records = dict()
    n_batches = int(math.ceil(len(table) / args.batch_size))
    vector_dimension = None
    for batch in tqdm(table.search().to_batches(batch_size=args.batch_size), total=n_batches):
        df = batch.to_pandas() # this helps conversion to numpy arrays and preserves the precision of the vectors
        for record in df.to_dict(orient="records"):
            word = record["word"]
            word_class = record["label"]
            vector = np.array(record["vector"], dtype=np.float32)
            if vector_dimension is None:
                vector_dimension = vector.shape[0]
            if word not in records:
                records[word] = {"class": set([word_class]), "vector": vector, "count": 1}
            else:
                records[word]["vector"] += vector
                records[word]["class"].add(word_class)
                records[word]["count"] += 1
    print(len(records), "unique words found.")
    

    vector_type = pa.list_(pa.float32(), vector_dimension)
    target_table_name = f"{table_name}_aggregated"
    schema = pa.schema(
                    [
                        pa.field('id', pa.int64()),
                        pa.field('label', pa.int8()),
                        pa.field('word', pa.string()),
                        pa.field('vector', vector_type)
                    ]
                )
    target_table = db.create_table(target_table_name, schema=schema, mode='overwrite')
    record_batch = []
    for i, (w, record) in enumerate(tqdm(records.items(), total=len(records))):
        if len(record["class"]) > 1:
            raise ValueError(f"Multiple classes found for word {w}. This should not happen, please check the data. Classes found: {record['class']}")
        record = {"id": i, "word": w, "vector": (record["vector"]/record["count"]), "label": int(list(record["class"])[0])}
        record_batch.append(record)
        if len(record_batch) >= args.batch_size:
            target_table.add(record_batch, on_bad_vectors='drop')
            record_batch = []
    if len(record_batch) > 0:
        target_table.add(record_batch, on_bad_vectors='drop')
    # target_table = None
    # records = []
    # for i, w in enumerate(tqdm(reversed(sorted(all_words.keys())), total=len(all_words))):
    #     vector = None
    #     label = None
    #     n_vectors = 0
    #     #safe_w = json.dumps(w, ensure_ascii=False)  # This will add quotes around the word and escape any special characters, making it safe for the query
    #     for batch in table.search().where(f'word = \'{w}\'').select(["vector", "label"]).to_batches():
    #         df_vectors = batch.to_pandas()  # This preserves the precision of the vectors
    #         for row in df_vectors.to_dict(orient="records"):
    #             v = row["vector"]
    #             l = row['label']
                
    #             if label is None:
    #                 label = l
    #             if label != l:
    #                 raise ValueError(f"Multiple classes found for word {w}. This should not happen, please check the data. Classes found: {label} and {l}")

    #             if vector is None:
    #                 vector = np.copy(v)
    #                 n_vectors = 1
    #             else:
    #                 vector += np.array(v)
    #                 n_vectors += 1
    #     if vector is not None:
    #         record = {"id": i, "word": w, "vector": (vector/n_vectors).tolist(), "label": int(label)}
    #         records.append(record)
    #     if len(records) >= args.batch_size:
    #         if target_table is None:
    #             v = records[0]["vector"]
    #             vector_dimension = len(v)
    #             vector_type = pa.list_(pa.float16(), vector_dimension)
    #             target_table_name = f"{table_name}_aggregated"
    #             schema = pa.schema(
    #                             [
    #                                 pa.field('id', pa.int64()),
    #                                 pa.field('label', pa.int8()),
    #                                 pa.field('word', pa.string()),
    #                                 pa.field('vector', vector_type)
    #                             ]
    #                         )
    #             target_table = db.create_table(target_table_name, schema=schema, mode='overwrite')
    #         table.add(records, on_bad_vectors='drop')
    #         records = []
    # if target_table is None:
    #     v = records[0]["vector"]
    #     vector_dimension = len(v)
    #     vector_type = pa.list_(pa.float16(), vector_dimension)
    #     target_table_name = f"{table_name}_aggregated"
    #     schema = pa.schema(
    #                     [
    #                         pa.field('id', pa.int64()),
    #                         pa.field('label', pa.int8()),
    #                         pa.field('word', pa.string()),
    #                         pa.field('vector', vector_type)
    #                     ]
    #                 )
    #     target_table = db.create_table(target_table_name, schema=schema, mode='overwrite')
    # table.add(records, on_bad_vectors='drop')
            
        


    

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()