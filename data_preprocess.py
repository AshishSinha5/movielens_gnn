import os
import math
import yaml
import numpy as np
import pandas as pd 
from datetime import datetime
import argparse

def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)
    return config

def temporal_split(df : pd.DataFrame, 
                   time_col : str,
                   train_start_time : datetime,
                   train_end_time : datetime,
                   val_start_time : datetime,
                   val_end_time : datetime,
                   test_start_time : datetime,
                   test_end_time : datetime):
    """
    """
    assert train_start_time < train_end_time < val_start_time < val_end_time < test_start_time < test_end_time
    train_df = df[(df[time_col] >= train_start_time) & (df[time_col] <= train_end_time)]
    val_df = df[(df[time_col] >= val_start_time) & (df[time_col] <= val_end_time)]
    test_df = df[(df[time_col] >= test_start_time) & (df[time_col] <= val_end_time)]

    return train_df, val_df, test_df


def get_timestamps(df : pd.DataFrame,
                   date_col : str,
                   train_prop : float,
                   val_prop : float,
                   test_prop : float):
    assert train_prop + val_prop + test_prop == 1
    num_samples = len(df)
    train_start_idx = 0
    train_end_idx = train_start_idx + math.floor(num_samples*train_prop)
    val_start_idx = train_end_idx + 1
    val_end_idx = val_start_idx + math.floor(num_samples*val_prop)
    test_start_idx = val_end_idx + 1
    test_end_idx = test_start_idx + math.floor(num_samples*test_prop) - 1 

    train_start_time = df[date_col].iloc[train_start_idx]
    train_end_time = df[date_col].iloc[train_end_idx]
    val_start_time = df[date_col].iloc[val_start_idx]
    val_end_time = df[date_col].iloc[val_end_idx]
    test_start_time = df[date_col].iloc[test_start_idx]
    test_end_time = df[date_col].iloc[test_end_idx]

    return train_start_time, train_end_time, val_start_time, val_end_time, test_start_time, test_end_time

def preprocess_data(users : pd.DataFrame,
               movies : pd.DataFrame,
               ratings : pd.DataFrame):
    ratings['Timestamp'] = pd.to_datetime(ratings['Timestamp'], unit='s')
    ratings = ratings.sort_values(by = 'Timestamp').reset_index()

    return users, movies, ratings

def create_data(config_path : str):
    config = load_config(config_path)
    root_data_path = config['root_data_path']
    users_data_path = os.path.join(root_data_path, 'users.dat')
    movies_data_path = os.path.join(root_data_path, 'movies.dat')
    ratings_data_path = os.path.join(root_data_path, 'ratings.dat')
    train_prop = config['train_prop']
    val_prop = config['val_prop']
    test_prop = config['test_prop']
    time_col = config['time_col']
    # Read users
    users = pd.read_csv(users_data_path, 
                        sep="::", 
                        engine="python", 
                        encoding="ISO-8859-1", 
                        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
    print(users.head())
    # Read movies
    movies = pd.read_csv(movies_data_path, 
                         sep="::", 
                         engine="python", 
                         encoding="ISO-8859-1", 
                         names=["MovieID", "Title", "Genres"])
    print(movies.head())
    # Read ratings
    ratings = pd.read_csv(ratings_data_path, 
                          sep="::", 
                          engine="python", 
                          encoding="ISO-8859-1", 
                          names=["UserID", "MovieID", "Rating", "Timestamp"])
    print(ratings.head())
    users, movies, ratings = preprocess_data(users, movies, ratings)

    train_start_time, train_end_time, val_start_time, val_end_time, test_start_time, test_end_time = get_timestamps(ratings, 
                                                                                                                     time_col,
                                                                                                                     train_prop,
                                                                                                                     val_prop,
                                                                                                                     test_prop)
    
    train_ratings, val_ratings, test_ratings = temporal_split(ratings, 
                                                              time_col,
                                                              train_start_time,
                                                              train_end_time,
                                                              val_start_time,
                                                              val_end_time,
                                                              test_start_time,
                                                              test_end_time)
    
    return users, movies, train_ratings, val_ratings, test_ratings



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='config.yaml', type=str)
    args = parser.parse_args()
    config_path = args.config_path
    create_data(config_path)