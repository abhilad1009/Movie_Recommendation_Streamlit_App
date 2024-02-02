# recommendation_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import data_loader

def collaborative_filter(data):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

    sim_options = {'name': 'cosine', 'user_based': False}
    cf_model = KNNBasic(sim_options=sim_options)
    cf_model.fit(trainset)

    return cf_model


def content_recommendations(user_id, N=5):
    # Get user's rated movies
    movies_with_genres = data_loader.genre_pivot(movies)
    user_rated_movies = train_data[train_data['userId'] == user_id].reset_index(drop=True)
    user_movie_ids = movies[movies['movieId'].isin(user_rated_movies['movieId'])]
    user_rated_movies = pd.merge(user_movie_ids, user_rated_movies)
    # print(user_rated_movies)

    user_rated_movies = user_rated_movies[['movieId','rating']]
    user_genres = movies_with_genres[movies_with_genres.movieId.isin(user_rated_movies.movieId)]
    user_genres.reset_index(drop=True, inplace=True)

    # Next, let's drop redundant columns
    user_genres.drop(['movieId','title','genres'], axis=1, inplace=True)
    # user_rated_movies
    # Let's view chamges
    # print('Shape of user_rated_movies is:',user_rated_movies.shape)
    # print('Shape of user_genres is:',user_genres.shape)
    user_profile = user_genres.T.dot(user_rated_movies.rating)
    # print(user_profile)

    movies_with_genres = data_loader.genre_pivot(movies[(~movies['movieId'].isin(user_rated_movies.movieId.values))])
    movies_with_genres = movies_with_genres.set_index(movies_with_genres.movieId)
    movies_with_genres.drop(['movieId','title','genres'], axis=1, inplace=True)
    # print(movies_with_genres)

    recommendation_table = (movies_with_genres.dot(user_profile)) / user_profile.sum()
    recommendation_table.sort_values(ascending=False, inplace=True)
    # print(recommendation_table)

    copy = movies.copy(deep=True)
    copy = copy.set_index('movieId', drop=True)

    top_N_index = recommendation_table.index[:N].tolist()
    # finally we slice these indices from the copied movies df and save in a variable
    recommended_movies = copy.loc[top_N_index, :]

    return recommended_movies

def rescale(data):
    X = data.copy()
    X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

    scaler = MinMaxScaler(feature_range = (0.5, 5))
    scaler.fit(X)
    pred_n = scaler.transform(X)
    return pred_n

def cf_recommendation(user_id,train_data, movies,N = 5):
    user_data = train_data.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

    dummy_train = train_data.copy()
    dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
    dummy_train = dummy_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(1)

    user_similarity = cosine_similarity(user_data)
    # print(user_similarity.shape)
    user_similarity[np.isnan(user_similarity)] = 0
    user_predicted_ratings = np.dot(user_similarity, user_data)
    user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)
    movie_ids = user_data.columns.tolist()

    pred_n = rescale(user_final_ratings)
    # print(pred_n.shape)

    movie_ratings = np.nan_to_num(pred_n[user_id-1]).tolist()
    pred_n= sorted(zip(movie_ratings,movie_ids),key=lambda x: x[0],reverse=True)[:N]
    l = []
    for i in pred_n:
        row = movies.loc[movies['movieId'] == i[1]]
        tmp = [row['movieId'].values[0],row['title'].values[0],row['genres'].values[0],i[0]]
        l.append(tmp)
    pred_n = pd.DataFrame(l,columns=['movieId','title','genres','rating'])
    return pred_n


def test_global_error(train_data, test_data):
    dummy_test = test_data.copy()
    dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)
    dummy_test = dummy_test.pivot(index ='userId', columns = 'movieId', values = 'rating').fillna(0)

    test_user_features = test_data.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
    user_data = train_data.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
    test_user_similarity = cosine_similarity(user_data)
    # print(test_user_similarity.shape)
    test_user_similarity[np.isnan(test_user_similarity)] = 0
    user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
    test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)


    pred = rescale(test_user_final_rating)

    test = test_data.pivot(index = 'userId', columns = 'movieId', values = 'rating')
    total_non_nan = np.count_nonzero(~np.isnan(pred))
    tmp_test= test.to_numpy()
    coverage = []
    for i in range(len(tmp_test)):
        tmp1 = np.nan_to_num(tmp_test[i]).tolist()
        test_rating = [(tmp1[j],j) for j in range(len(tmp1))]
        tmp2 = np.nan_to_num(pred[i]).tolist()
        pred_rating = [(tmp2[j],j) for j in range(len(tmp2))]
        test_rating.sort(reverse=True)
        pred_rating.sort(reverse=True)
        predtop5 = set([i[1] for i in pred_rating[:5]])
        top10 = set([i[1] for i in test_rating[:10]])
        coverage.append(len(top10.intersection(predtop5))/5)

    mae = np.abs(pred - test).sum().sum()/total_non_nan
    return mae,np.mean(coverage)

def recall(user_id, reccs, test_data):
    test_movies = test_data[test_data['userId']==user_id]
    tmp = test_movies[test_movies['movieId'].isin(reccs.movieId.values)]
    return len(tmp)/5

def global_recall(train_data,test_data):
    recalls = [recall(user_id, cf_recommendation(user_id,train_data), test_data) for user_id in test_data['userId'].unique()]
    return np.mean(recalls)