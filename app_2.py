# app900.py

import streamlit as st
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    /* Adjust the sidebar position */
    .sidebar .sidebar-content {
        position: fixed;
        overflow-y: hidden; /* Hide vertical scrollbar */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Data
ratings_df = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')
original_movies = movies_data

# Function to remove the year from movie titles
def remove_year(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

# Apply the function to the 'title' column
original_movies['title'] = original_movies['title'].apply(remove_year)

# Collaborative Filtering Setup
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
# Split the data
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
# Build and train an SVD model
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
    # Exclude the movie itself
    sim_scores = sim_scores[1:r_c]
    movie_indices = [i[0] for i in sim_scores]
    return original_movies['title'].iloc[movie_indices]

# Hybrid recommendations function with item-based recommendations
def hybrid_recommendations(userId, title, k=10, r_c=11):
    # Get content-based recommendations
    content_based_recs = get_content_based_recommendations(title, r_c)

    # Get collaborative filtering recommendations using item-based approach
    collab_filtering_recs = []
    try:
        movie_id = original_movies['movieId'][original_movies['title'] == title].values[0]
        user_id = userId
        # Calculate item-item similarity using item ratings
        item_similarity = cosine_sim.dot(ratings_df.T)
        # Get similar items to the target movie
        similar_items = item_similarity[movie_id]
        # Sort by similarity and get the top k
        top_similar_items = similar_items.argsort()[::-1][:k]
        collab_filtering_recs = [original_movies['title'].iloc[i] for i in top_similar_items]
    except:
        pass

    recommendations = list(set(content_based_recs) | set(collab_filtering_recs))

    return recommendations

# Streamlit interface
st.title("Movie Recommendation System")
user_id = st.text_input("Select User ID:", )
movie_title = st.text_input("Enter Movie Title:")
num_recommendations = st.text_input("Number of Recommendations:", )

if st.button("Get Recommendations"):
    recommendations = hybrid_recommendations(user_id, movie_title, num_recommendations)

    st.subheader("Recommended Movies:")
    for i, movie in enumerate(recommendations):
        # Add a border around each recommended movie
        st.markdown(
            f"<div style='border: 2px solid #ccc; padding: 10px; margin: 5px;'>{i+1}. {movie}</div>",
            unsafe_allow_html=True
        )

# Sidebar images
caption1 = "Toy Story"
image_url1 = "https://media.comicbook.com/2019/03/toy-story-4-poster-1163565.jpeg"  # Replace with your image URL
st.sidebar.image(image_url1, caption=caption1, use_column_width=True)

caption2 = "Mulan"
image_url2 = "https://media.wdwnt.com/2019/12/EK9zDtnUcAA5zXM-4.jpeg"  # Replace with your image URL
st.sidebar.image(image_url2, caption=caption2, use_column_width=True)

caption3 = "The Johnsons"
image_url3 = "https://intheposter.com/cdn/shop/products/the-family-comedy-in-the-poster-1_1200x.jpg?v=1694762497"  # Replace with your image URL
st.sidebar.image(image_url3, caption=caption3, use_column_width=True)
