import argparse
from pathlib import Path
import json
import os.path
import shutil

import torch
from tqdm import trange, tqdm

from custom.nlp_models import XLMRobertaModel

def main():
    parser = argparse.ArgumentParser(description="Train the local version of the XLMRoberta model for federated health")
    #parser.add_argument('model_path', help="Directory to read the model from", type=Path)
    parser.add_argument('nvflare_config_dir', help="Directory to read configurations from. This should contain the config_fed_server.json and config_fed_client.json used for training the federated model which we use to pick up number of rounds etc.", type=Path)
    parser.add_argument('data_prefix', help="The prefix used to construct the dataset paths. Often something like 'site-1' or 'site-2'")
    parser.add_argument('--app-dir', help="Directory to save training output to", type=Path, default=Path("local_training"))
    
    args = parser.parse_args()
    app_dir = args.app_dir / args.data_prefix
    app_dir.mkdir(exists_ok=True, parents=True)
    
    server_config_path = args.nvflare_config_dir / "config_fed_server.json"
    client_config_path = args.nvflare_config_dir / "config_fed_client.json"
    
    # Extract server arguments
    with open(server_config_path) as fp:
        server_fed_config = json.load(fp)
    max_epochs = server_fed_config["num_rounds"]
        
    # Extract client arguments
    with open(client_config_path) as fp:
        client_fed_config = json.load(fp)
    
    model_path = None
    config_path = None
    data_path = None
    
    # This is hardcoded to work with the jobs definition we have made.
    for component in client_fed_config["components"]:
        if component["path"] == "custom.learners.nlp_learner.NLPLearner":
            data_path = component["args"]["data_path"]
            model_path = component["args"]["model_path"]
            config_path = component["args"]["config_path"]
            
    if model_path is None:
        raise RuntimeError(f"Could not find a model config path in {args.config_path}. Does it have a PTFileModelPersistor component?")
    
    model = XLMRobertaModel(model_path, config_path)
    
    train_dataset_path = os.path.join(data_path, f"{args.data_prefix}_train.txt")
    dev_dataset_path = os.path.join(data_path, f"{args.data_prefix}_dev.txt")
    test_dataset_path = os.path.join(data_path, f"{args.data_prefix}_test.txt")
    model.initialize(app_dir, train_dataset_path, dev_dataset_path, test_dataset_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_perplexity = float("inf")
    best_model_path = None
    latest_model_path = app_dir / "latest_model_epoch0.pt"
    torch.save(model.state_dict(), latest_model_path)
    
    for epoch in trange(max_epochs, desc="Epoch"):
        model.train()
        for inner_epoch in range(model.aggregation_epochs):
            for batch_data in tqdm(model.train_dataloader, desc='Batch'):
                model.fit_batch(batch_data)
        
        perplexity = model.local_valid()
        
        if perplexity < best_perplexity:
            best_model_path = os.join(args.app_dir / f"best_model_epoch-{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            
        latest_model_path = os.join(args.app_dir / f"latest_model_epoch-{epoch+1}.pt")
        torch.save(model.state_dict(), latest_model_path)
                
            
    
    
if __name__ == '__main__':
    main()