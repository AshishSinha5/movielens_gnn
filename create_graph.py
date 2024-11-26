import os
import numpy as np
import pandas as pd 
import yaml 
import ast

from sklearn.preprocessing import LabelEncoder

import torch 
from torch_geometric.data import HeteroData

def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)
    return config

def create_graph(users, movies, train_ratings):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    users['UserID'] = user_encoder.fit_transform(users['UserID'])
    movies['MovieID'] = movie_encoder.fit_transform(movies['MovieID'])
    train_ratings['UserID'] = user_encoder.transform(train_ratings['UserID'])
    train_ratings['MovieID'] = movie_encoder.transform(train_ratings['MovieID'])
    train_pos_ratings = train_ratings[train_ratings['Rating'] >= 4]
    data = HeteroData()
    data['movie'].x = torch.tensor(movies['Genre_Embedding'].apply(lambda x : ast.literal_eval(x)) + movies['Title_Embedding'].apply(lambda x : ast.literal_eval(x)) + movies['Year'].apply(lambda x: [int(x)]))
    data['users'].x = torch.tensor(users[['Gender', 'Age', 'Occupation']].values)

    edge_index = train_pos_ratings[['UserID', 'MovieID']].values.T
    data['user', 'likes', 'movie'].edge_index = torch.tensor(edge_index)

    return data

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    users = pd.read_csv(os.path.join(config['transformed_data_root_path'], 'users.csv'))
    movies = pd.read_csv(os.path.join(config['transformed_data_root_path'], 'movies.csv'))
    train_ratings = pd.read_csv(os.path.join(config['transformed_data_root_path'], 'train_ratings.csv'))

    data = create_graph(users, movies, train_ratings)

    os.makedirs(config['graph_path'], exist_ok=True)
    torch.save(data, os.path.join(config['graph_path'], 'graph.pt'))