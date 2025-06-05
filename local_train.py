import argparse
from pathlib import Path
import json
import os.path
import shutil
import socket


import torch
from tqdm import trange, tqdm

import sys


APP_PATH = Path('apps')/ 'xlmroberta_mlm'
TRAIN_DATA_ENV = "FH_TRAINING_DATA"
DEV_DATA_ENV = "FH_DEV_DATA"
TEST_DATA_ENV = "FH_TEST_DATA"

from custom.nlp_models import XLMRobertaModel

def main():
    parser = argparse.ArgumentParser(description="Train the local version of the XLMRoberta model for federated health")
    #parser.add_argument('model_path', help="Directory to read the model from", type=Path)
    parser.add_argument('--app-dir', 
                        help="Directory where the NVFLARE app resides.", 
                        type=Path,
                        default=APP_PATH)
    parser.add_argument('--site', 
                        help="The identifier for the site, will mainly be used for organizing output.", 
                        default=socket.gethostname())
    parser.add_argument('--workspace-dir', 
                        help="Directory to save training output to", 
                        type=Path, 
                        default=Path("local_training"))
    
    args = parser.parse_args()
    workspace_dir = args.workspace_dir / args.site
    workspace_dir.mkdir(exist_ok=True, parents=True)
    
    server_config_path = args.app_dir / 'config' / 'config_fed_server.json'
    client_config_path = args.app_dir / 'config' / 'config_fed_client.json'
    
    # Extract server arguments
    with open(server_config_path) as fp:
        server_fed_config = json.load(fp)
    max_epochs = server_fed_config["num_rounds"]
        
    # Extract client arguments
    with open(client_config_path) as fp:
        client_fed_config = json.load(fp)
    
    config_name = None
    
    # This is hardcoded to work with the jobs definition we have made.
    for component in client_fed_config["components"]:
        if component["path"] == "custom.nlp_learner.NLPLearner":
            config_name = component["args"]["config_name"]
            
    if config_name is None:
        raise RuntimeError(f"Could not find a model config path in {args.config_path}. Does it have a PTFileModelPersistor component?")
    
    model = XLMRobertaModel(config_name)
    
    
    if TRAIN_DATA_ENV in os.environ:
        training_data_path = os.environ[TRAIN_DATA_ENV]
    else:
        raise RuntimeError(f"Environmental variable '{TRAIN_DATA_ENV}' not set, no training data path")
    if DEV_DATA_ENV in os.environ:
        dev_data_path = os.environ[DEV_DATA_ENV]
    else:
        raise RuntimeError(f"Environmental variable '{DEV_DATA_ENV}' not set, no dev data path")
    
    if TEST_DATA_ENV in os.environ:
        test_data_path = os.environ[TEST_DATA_ENV]
    else:
        raise RuntimeError(f"Environmental variable '{TEST_DATA_ENV}' not set, no test data path")
    
    model.initialize(workspace_dir, training_data_path, dev_data_path, test_data_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_perplexity = float("inf")
    best_model_path = None
    latest_model_path = workspace_dir / "latest_model_epoch0.pt"
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