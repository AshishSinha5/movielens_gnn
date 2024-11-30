import os 
import numpy as np
import pandas as pd
import yaml

import torch 
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn.functional import binary_cross_entropy_with_logits

def get_negative_samples(pos_edge_index, num_nodes, num_neg_samples, device):
    """
    Generate negative samples (edges that do not exist in the graph).
    Args:
        pos_edge_index (torch.Tensor): Positive edge index (2 x E_pos).
        num_nodes (int): Total number of nodes in the graph.
        num_neg_samples (int): Number of negative samples to generate.
        device (torch.device): Device to use for computations.
    Returns:
        torch.Tensor: Negative edge index (2 x num_neg_samples).
    """
    # Create a set of positive edges
    pos_edges = set(map(tuple, pos_edge_index.t().tolist()))

    # Generate random candidate edges
    neg_edges = set()
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
        dst = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
        candidates = set(zip(src.tolist(), dst.tolist()))
        # Exclude positive edges and self-loops
        neg_edges.update(candidates - pos_edges)

    # Convert to tensor
    neg_edge_index = torch.tensor(list(neg_edges), device=device).t()
    return neg_edge_index

def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)
    return config


config_path = 'config.yaml'
config = load_config(config_path)

data = torch.load(os.path.join(config['graph_path'], 'graph.pt'))

# Define a GNN model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "likes", "movie"].edge_label_index,
        )

        return pred
    
# Step 3: Define training procedure with negative sampling
def graphsage_loss(embeddings, pos_edge_index, neg_edge_index):
    """
    Compute GraphSAGE loss using dot product for positive and negative edges.
    Args:
        embeddings (torch.Tensor): Node embeddings.
        pos_edge_index (torch.Tensor): Positive edge index (2 x E_pos).
        neg_edge_index (torch.Tensor): Negative edge index (2 x E_neg).
    Returns:
        torch.Tensor: GraphSAGE loss value.
    """
    # Positive edge scores
    pos_scores = torch.sum(
        embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]], dim=1
    )  # Dot product
    # Negative edge scores
    neg_scores = torch.sum(
        embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]], dim=1
    )

    # Loss for positive edges
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()

    # Loss for negative edges
    neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()

    # Final loss (normalized by batch size)
    return pos_loss + neg_loss

# Step 4: Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGNN(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = data.to(device)
loader = LinkNeighborLoader(
    data=data,
    num_neighbors={key: [30, 30] for key in data.edge_types},
    edge_label_index=(('user', 'likes', 'movie'), data[('user', 'likes', 'movie')].edge_index),
    batch_size = 2048*2,
    shuffle=True
)

model.train()
for epoch in range(10):
    total_loss = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        # Forward pass
        embeddings = model(batch.x_dict, batch.edge_index_dict)

        # Positive edges
        pos_edge_index = batch[('user', 'likes', 'movie')].edge_index

        # Generate negative samples
        neg_edge_index = get_negative_samples(
            pos_edge_index, 
            num_nodes=embeddings['user'].size(0), 
            num_neg_samples=pos_edge_index.size(1), 
            device=device
        )
        
        # Compute loss
        loss = graphsage_loss(embeddings['user'], pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"{epoch = }, batch = {i}, {loss.item() = }")
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Step 5: Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x_dict, data.edge_index_dict)
    user_embeddings = embeddings['user']  # User embeddings
    movie_embeddings = embeddings['movie']  # Movie embeddings

print("User embeddings shape:", user_embeddings.shape)
print("Movie embeddings shape:", movie_embeddings.shape)

# save embeddings
os.makedirs(config['embeddings_root_path'], exist_ok=True)
torch.save(embeddings, os.path.join(config['embeddings_root_path'], 'embeddings.pts'))

# save model 
os.makedirs(config['model_path'], exist_ok=True)
torch.save(model.state_dict(), os.path.join(config['model_path'], 'model.pt'))