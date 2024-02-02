import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data():
    
    movies = pd.read_csv('data/movies.csv')

    movies['title'] = movies['title'].apply(lambda x: x.strip())
    movies['genres'] = movies.genres.str.split('|')
    movies.movieId = movies.movieId.astype('int32')

    ratings = pd.read_csv('data/ratings.csv')
    ratings.drop('timestamp', axis=1, inplace=True)

    merged_data = pd.merge(movies,ratings,on='movieId')
    return movies, ratings, merged_data

def split_data(data):

    train_data, test_data = train_test_split(data, test_size=0.5, stratify=data['userId'])

    return train_data, test_data

def genre_pivot(data):
    movies_with_genres = data.copy(deep=True)
    x = []
    for index, row in data.iterrows():
        x.append(index)
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
    movies_with_genres = movies_with_genres.fillna(0)

    return movies_with_genres

