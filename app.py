# Import necessary libraries
import streamlit as st
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Function to remove the year from movie titles
def remove_year(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

# Load Data
ratings_df = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')
original_movies = movies_data
original_movies['title'] = original_movies['title'].apply(remove_year)

# Collaborative Filtering Setup
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD()
svd.fit(trainset)

# Content-Based Setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(original_movies['genres'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get content-based recommendations
def get_content_based_recommendations(title, r_c=11):
    idx = original_movies.index[original_movies['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:r_c]
    movie_indices = [i[0] for i in sim_scores]
    return original_movies['title'].iloc[movie_indices]

# Hybrid recommendations function with item-based recommendations
def hybrid_recommendations(userId, title, k=10, r_c=11):
    content_based_recs = get_content_based_recommendations(title, r_c)
    collab_filtering_recs = []

    try:
        movie_id = original_movies['movieId'][original_movies['title'] == title].values[0]
        user_id = userId
        item_similarity = cosine_sim.dot(ratings_df.T)
        similar_items = item_similarity[movie_id]
        top_similar_items = similar_items.argsort()[::-1][:k]
        collab_filtering_recs = [original_movies['title'].iloc[i] for i in top_similar_items]
    except:
        pass

    recommendations = list(set(content_based_recs) | set(collab_filtering_recs))
    return recommendations

# Streamlit interface
def main():
    st.title("Movie Recommendation System")
    user_id = st.text_input("Select User ID:")
    movie_title = st.text_input("Enter Movie Title:")
    num_recommendations = st.text_input("Number of Recommendations:")

    if st.button("Get Recommendations"):
        recommendations = hybrid_recommendations(user_id, movie_title, num_recommendations)
        st.subheader("Recommended Movies:")

        for i, movie in enumerate(recommendations):
            button = st.checkbox(f"Show Explanation for {movie}", False)
            if button:
                explanation = generate_explanation(movie, movie_title)  # Generate an explanation
                st.markdown(
                    f"""<div style='border: 2px solid #ccc; padding: 10px; margin: 5px;'>{i+1}. {movie}<br>Type of Movie is {movie_genre}<br>Explanation: {explanation}</div>""",
                    unsafe_allow_html=True
                )

# Function to generate an explanation
def generate_explanation(movie_title, user_preferences):
    explanation = f"We recommend '{movie_title}' because it matches your preference for {user_preferences} and It belongs to the genre {original_movies[original_movies['title'] == movie_title]['genres'].values[0]}."
    return explanation

if __name__ == "__main__":
    main()
