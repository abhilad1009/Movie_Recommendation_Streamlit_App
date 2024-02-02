import streamlit as st
import streamlit_authenticator as stauth
import data_loader, recommendation_model
import numpy as np
import pandas as pd

import yaml
from yaml.loader import SafeLoader

with open('auth.yaml') as file:
    auth = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    auth['credentials'],
    auth['cookie']['name'],
    auth['cookie']['key'],
    auth['cookie']['expiry_days'],
    auth['preauthorized']
)

# Load and preprocess data
@st.cache_data
def load_data_main():
    movies, ratings, merged_data = data_loader.load_data()
    train_data, test_data = data_loader.split_data(ratings)
    return movies, ratings, merged_data, train_data, test_data
# movies, ratings, merged_data = data_loader.load_data()
# # enhanced_data = data_loader.enhance_features(merged_data)

# train_data, test_data = data_loader.split_data(ratings)

# movies_with_genres_train = data_loader.genre_pivot(movies[movies['movieId'].isin()])
# movies_with_genres_test = data_loader.genre_pivot(movies)




# Streamlit app pages
def data_overview():
    st.header("Data Overview")
    st.dataframe(merged_data.head())
    st.write("No of users: ",len(ratings['userId'].unique()))
    st.write("No of movies: ",len(movies['movieId'].unique()))
    st.write("No of user ratings: ",len(ratings))
    tmp = ratings.groupby(['movieId'])['rating'].mean()
    st.write("Plot of average rating of first 5 movies")
    st.pyplot(tmp.head(5).plot.barh(stacked=True).figure)

def feature_enhancement_page():
    st.header("Feature Enhancement Overview")
    st.write("For content based filtering, we separate the genres in a list of values and then pivot them to convert into movie x genre table")
    st.write("For collaborative filtering model, we pivot the data into user x movie format")
    st.code('''user_data = train_data.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)''',language='python')
    st.write("We also create dummy train and test set to identify movies not rated and rated by user.")
    st.code( '''dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)''',language='python')
    st.write("We also rescale the predicted ratings")
    st.code('''def rescale(data):
    X = data.copy()
    X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

    scaler = MinMaxScaler(feature_range = (0.5, 5))
    scaler.fit(X)
    pred_n = scaler.transform(X)
    return pred_n''',language='python')


def test_train_split_overview():
    st.header("Test Train Split Overview")
    code = '''def split_data(data):

    train_data, test_data = train_test_split(data, test_size=0.5, stratify=data['userId'])

    return train_data, test_data'''
    st.code(code, language='python')
    st.write(" Since we want movies review by a user to be equally distributed in train and test set, we use stratify by 'userId' argument while splitting the dataset")
    st.write(f"#### Training Data Shape: {train_data.shape}")
    st.write(f"####  Testing Data Shape: {test_data.shape}")

def recommendation_abstract():
    st.header("Recommendation Abstract")
    st.write("I tried 2 different approaches: 1) Content based filtering and 2) Collaborative filtering")
    st.write("Content based filtering suffered from generic recommendations for all user, thus not providing personalization.")
    st.write("In Collaborative filtering, I implemented user based filtering. In this approach, each user is presented as a vector of movies that they liked. We than calculate similarity matrix of user vectors and multiply it with user ratings to generate weighted ratings. We then rescale the weighted ratings back to 0-5 and select the top 5 rated movies for each user.")

    st.header("Conclusion")
    st.write("When calculating the predicted ratings of movies which are rated by user, the algorithm has a mean absolute error of ~1.2, with that skew being on lower side. This is respectable but could be improved.")
    st.write("We also calculate the global coverage, where we find the percent of top 5 predictions among rated movies to be present in top 10 of the actual rated movies. We get a global coverage of >0.5, i.e. more than 50% of the time, the algorithm rates the movie in correct tier.")
    st.write("However, in case of direct predictions of top 5 movies, the recall for each user is close to 0. This is expected as we are not considering genre information and there is inherent randomness to every user.")

def recommendation_demo():
    st.header("Recommendation Demo")
    user_id = int(st.text_input("Enter User ID:",value = 32))
    user_data = train_data[train_data['userId'] == user_id].reset_index(drop=True)

    if not user_data.empty:
        user_movie_ids = movies[movies['movieId'].isin(user_data['movieId'])]
        user_data_train = pd.merge(user_movie_ids, user_data)
        st.subheader(f"Movies from the training data set for User {user_id}")
        st.dataframe(user_data_train.sort_values('rating',ascending=False))

        st.subheader("Top 5 Movie Recommendations and Predicted Ratings")

        top_recommendations = recommendation_model.cf_recommendation(user_id,train_data, movies,N = 5)

        st.dataframe(top_recommendations)

        st.subheader("Model Evaluation on Test Data")
        mae, coverage = recommendation_model.test_global_error(train_data,test_data)
        st.write("####   Mean absolute error: ",mae)
        st.write("Mean absolute error indicates the error between the predicted ratings and actual ratings of movies rated by user in test set")
        st.write("#### Global coverage: ",coverage)
        st.write("Global coverage indicates the ratio of top 5 predicted (known rated) movies present in top 10 movies rated by user in the test set")
        st.write("#### User recall: ",recommendation_model.recall(user_id,top_recommendations,test_data))
        st.write("Recall for each user is the ratio of top 5 predicted (unknown) movies present in test set for that user")
        # st.write("User recall: ",recommendation_model.global_recall(train_data,test_data))

        st.subheader(f"Movies from the test data set for User {user_id}")
        user_data = test_data[test_data['userId'] == user_id].reset_index(drop=True)
        user_movie_ids = movies[movies['movieId'].isin(user_data['movieId'])]
        user_data_test= pd.merge(user_movie_ids, user_data)
        st.dataframe(user_data_test.sort_values('rating',ascending=False))


    else:
        st.warning("User not found.")

# Streamlit app routing
def main():
    st.title('Movie Recommendation App')
    name, authentication_status, username = authenticator.login('main', fields={'Form name':'Login', 'Username':'Username', 'Password':'Password', 'Login':'Login'})
    if st.session_state["authentication_status"]:
        st.write(f'Welcome *{st.session_state["name"]}*')
        page = st.sidebar.selectbox("Select a page", ["Data Overview", "Feature Enhancement",
                                                  "Test Train Split Overview", "Recommendation Abstract",
                                                  "Recommendation Demo"])

        global movies, ratings, merged_data, train_data, test_data

        movies, ratings, merged_data, train_data, test_data = load_data_main()

        if page == "Data Overview":
            data_overview()
        elif page == "Feature Enhancement":
            feature_enhancement_page()
        elif page == "Test Train Split Overview":
            test_train_split_overview()
        elif page == "Recommendation Abstract":
            recommendation_abstract()
        elif page == "Recommendation Demo":
            recommendation_demo()

        authenticator.logout('Logout', 'sidebar')
    elif  st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif  st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
        load_data_main.clear()

    
if __name__ == '__main__':
    main()
