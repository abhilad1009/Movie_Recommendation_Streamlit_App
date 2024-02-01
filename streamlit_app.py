# streamlit_app.py
import streamlit as st
import data_loader, data_loader, recommendation_model

# Load and preprocess data
raw_data = data_loader.load_data()
enhanced_data = data_loader.enhance_features(raw_data)

# # Split data
train_data, test_data = data_loader.split_data(enhanced_data)

# # Train recommendation model
# model = recommendation_model.train_model(train_data)

# Streamlit app pages
def data_overview():
    st.header("Data Overview")
    st.dataframe(raw_data.head())

def feature_enhancement_page():
    st.header("Feature Enhancement Overview")
    st.dataframe(enhanced_data.head())

def test_train_split_overview():
    st.header("Test Train Split Overview")
    st.write(f"Training Data Shape: {train_data.shape}")
    st.write(f"Testing Data Shape: {test_data.shape}")

def recommendation_abstract():
    st.header("Recommendation Abstract")
    st.write("We used a Random Forest Regressor for movie rating prediction.")
    # Add more details as needed

def recommendation_demo():
    st.header("Recommendation Demo")
    user_id = st.text_input("Enter User ID:",value = 1)
    user_data = enhanced_data[enhanced_data['userId'] == int(user_id)]

    if not user_data.empty:
        st.subheader(f"Movies from the training data set for User {user_id}")
        st.dataframe(user_data)

        st.subheader("Top 5 Movie Recommendations and Predicted Ratings")
    #     # Implement recommendation logic and display results

    #     st.subheader("Model Evaluation on Test Data")
    #     evaluation_results = recommendation_model.evaluate_model(model, test_data)
    #     st.write(evaluation_results)
    else:
        st.warning("User not found.")

# Streamlit app routing
def main():
    st.title('Movie Recommendation App')

    page = st.sidebar.selectbox("Select a page", ["Data Overview", "Feature Enhancement",
                                                  "Test Train Split Overview", "Recommendation Abstract",
                                                  "Recommendation Demo"])

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

if __name__ == '__main__':
    main()
