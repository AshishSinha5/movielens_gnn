import os 
import numpy as np
import pandas as pd
import yaml

import torch 
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)
    return config


config_path = 'config.yaml'
config = load_config(config_path)

data = torch.load(os.path.join(config['graph_path'], 'graph.pt'))


loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors={key: [3] * 2 for key in data.edge_types},
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    input_nodes=('users', [1,2])
)

for mini_batch in loader:
    print(mini_batch)