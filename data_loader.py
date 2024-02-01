# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data():
    # Load data from CSV or Kaggle dataset
    path = os.getcwd()
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    data = pd.merge(movies,ratings,on='movieId')
    return data

def split_data(data):
    # Implement user-based splitting logic
    train_data, test_data = train_test_split(data, test_size=0.5, stratify=data['userId'])
    return train_data, test_data

# feature_enhancement.py
def enhance_features(data):
    # Example: Create a new feature 'WatchedBefore' based on historical data
    # data['WatchedBefore'] = data.groupby('MovieID')['Rating'].transform('count') > 1
    return data
