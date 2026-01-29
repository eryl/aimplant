import argparse
from pathlib import Path
import math

import h5py
from tqdm import trange, tqdm
import numpy as np
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


def main():
    parser = argparse.ArgumentParser(description="Take precalculated vectors and populate vector database with them")
    parser.add_argument("vector_storage", help="HDF5 file containing calculated vectors", type=Path)
    parser.add_argument("--output-dir", help="Where to save the vector database", type=Path)
    parser.add_argument("--layers", help="What layers to include in the representation", nargs='+', default=('-1', '-2', '-3', '-4'))
    parser.add_argument("--aggregation", help="Method of aggregating vectors", choices=("mean", "sum"), default="mean")
    parser.add_argument("--chunk-size", help="How many vectors to read at a time", type=int, default=512)
    
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else args.vector_storage.parent
    output_dir.mkdir(parents=True, exist_ok=True)
        # We need to check the model format for the federated training
    uri = output_dir / "lancedb"
    table = None
    db = lancedb.connect(uri)
    with h5py.File(args.vector_storage, mode='r') as store:
        n_vectors = len(store['words'])  # There are as many vectors as words
        n_chunks = int(math.ceil(n_vectors / args.chunk_size))
        layer_datasets = [dataset for dataset in store['vectors'].values() if str(dataset.attrs['layer_idx']) in args.layers]
        for i in trange(n_chunks, desc="Processing chunks"):
            vectors = []
            start = i * args.chunk_size
            end = start + args.chunk_size
            words = store['words'][start:end]
            for layer_ds in layer_datasets:
                layer_vectors = layer_ds[start:end]
                vectors.append(layer_vectors)
            if args.aggregation == 'mean':
                aggregated_vectors = np.mean(vectors, axis=0)
            elif args.aggregation == 'sum':
                aggregated_vectors = np.sum(vectors, axis=0)
            else:
                raise RuntimeError(f"Unknown aggregation method {args.aggretation}")
            data = [{'word': str(word, encoding='utf-8'), 'vector': vector} for word, vector in zip(words, aggregated_vectors)]
            if table is None:
                table = db.create_table("words", data)
            else:
                table.add(data)
    if table is not None:
        print("Building index")
        table.create_index(metric="cosine", num_partitions=16, num_sub_vectors=4, vector_column_name='vector')




if __name__ == '__main__':
    main()
