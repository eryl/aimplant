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

from federatedhealth.nlp_models import XLMRobertaModel
from federatedhealth.config import load_config

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
    
    model = XLMRobertaModel()
    
    config = load_config()
    training_data_path = config.data_config.training_data
    dev_data_path = config.data_config.dev_data
    test_data_path = config.data_config.test_data
    
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
            best_model_path = workspace_dir / f"best_model_epoch-{epoch+1}.pt"
            torch.save(model.state_dict(), best_model_path)
            
        latest_model_path = workspace_dir / f"latest_model_epoch-{epoch+1}.pt"
        torch.save(model.state_dict(), latest_model_path)
                
            
    
    
if __name__ == '__main__':
    main()