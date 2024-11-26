import os 
import re
import argparse
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopword_tokens

from data_preprocess import create_data

w2v_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300_2.bin', binary=True)

def compute_average_embedding(genres, w2v_model):
    embeddings = [w2v_model.get_vector(genre) for genre in genres if genre in w2v_model.index_to_key]
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()
    else:
        # Return a zero vector if no genres are found in the model
        return np.zeros(w2v_model.vector_size).tolist()
    
def preprocess_genre(genre):
    genre = genre.lower()
    genre = genre.split('|')
    return genre
    
def preprocess_title(title):
    title = title.lower()
    title = title.split(' ')
    title = remove_stopword_tokens(title)
    title  = [re.sub(r"[^ a-zA-Z0-9]+",'',word) for word in title]
    title = [word.strip() for word in title]
    title = [word for word in title if len(word)]
    return title

def feat_eng(users, movies):
    users['Gender'] = users['Gender'].map({'M' : 1, 'F' : 0})
    # Regex to extract title and year
    movies[['Title', 'Year']] = movies['Title'].str.extract(r'^(.*?)(?: \((\d{4})\))?$')
    movies['Year'].fillna(0, inplace = True)
    movies['Genre_List'] = movies['Genres'].apply(preprocess_genre)
    movies['Genre_Embedding'] = movies['Genre_List'].apply(lambda x: compute_average_embedding(x, w2v_model))
    movies['Title_List'] = movies['Title'].apply(preprocess_title)
    movies['Title_Embedding'] = movies['Title_List'].apply(lambda x: compute_average_embedding(x, w2v_model))
    return users, movies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='config.yaml', type=str)
    args = parser.parse_args()
    config_path = args.config_path
    users, movies, train_ratings, val_ratings, test_ratings, config = create_data(config_path)
    users, movies = feat_eng(users, movies)

    os.makedirs(config['transformed_data_root_path'], exist_ok=True)
    users.to_csv(os.path.join(config['transformed_data_root_path'], 'users.csv'), index=False)
    movies.to_csv(os.path.join(config['transformed_data_root_path'], 'movies.csv'), index = False)
    train_ratings.to_csv(os.path.join(config['transformed_data_root_path'], 'train_ratings.csv'), index = False)
    val_ratings.to_csv(os.path.join(config['transformed_data_root_path'], 'val_ratings.csv'), index = False)
    test_ratings.to_csv(os.path.join(config['transformed_data_root_path'], 'test_ratings.csv'), index = False)
