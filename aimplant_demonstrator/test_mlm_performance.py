import argparse
from pathlib import Path
import json
import csv
from typing import Optional

import torch
from tqdm import trange, tqdm

from federatedhealth.nlp_models import XLMRobertaModel, load_model_from_checkpoint
from federatedhealth.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained XLMRoberta models on the test data")
    parser.add_argument('app_dir', 
                        help="App dir from training (either federated or local)", 
                        type=Path)
    parser.add_argument('--test-data', 
                        help="Path to text file with test data, if not given, uses the one from config", 
                        type=Path)
    parser.add_argument('--model-filter', help="Give a glob filter to only include matching files", default="*.pt")
    args = parser.parse_args()
    
    config = load_config()
    training_data_path = config.data_config.training_data
    dev_data_path = config.data_config.dev_data
    test_data_path = config.data_config.test_data

    config.training_args.eval_samples = None  # Evaluate on full dev/test set
    if args.test_data:
        test_data_path = args.test_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    output_dir = args.app_dir / "local_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path: Optional[Path] = None
    best_model_performance = float('inf')
    with open(output_dir / "test_performance.csv", 'w', newline='') as csvfile:
        fieldnames = ['model_checkpoint', 'dev_loss', 'dev_perplexity', 'test_loss', 'test_perplexity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  
        # We need to check the model format for the federated training
        models: list[Path] = sorted(args.app_dir.glob(f"**/{args.model_filter}"))
        for model_path in tqdm(models, desc="Evaluating models"):
            if model_path.is_symlink():
                continue
            model = load_model_from_checkpoint(model_path)
            model.initialize(args.app_dir, training_data_path, dev_data_path, test_data_path, training_override=config.training_args)
            model.to(device)
            dev_loss, dev_perplexity = model.local_valid()
            test_loss, test_perplexity = model.local_test()
            performance_output_file = output_dir / f"{model_path.stem}_performance.json"
            writer.writerow({'model_checkpoint': model_path.name,
                             'dev_loss': dev_loss,
                             'dev_perplexity': dev_perplexity,
                             'test_loss': test_loss,
                             'test_perplexity': test_perplexity})
            csvfile.flush()
            if dev_perplexity < best_model_performance:
                best_model_path = model_path
            with open(performance_output_file, 'w') as f:
                json.dump({"dev_loss": dev_loss, "dev_perplexity": dev_perplexity,
                           "test_loss": test_loss, "test_perplexity": test_perplexity}, 
                        f,
                        indent=2)

    if best_model_path is not None:
        best_model_symlink: Path = output_dir / "best_local_model.pt"
        best_model_symlink.unlink(missing_ok=True)
        best_model_symlink.symlink_to(best_model_path.absolute())


if __name__ == '__main__':
    main()