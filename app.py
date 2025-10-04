import streamlit as st
import pickle
import pandas as pd
import requests
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ----------------------------
# Download Large File from Google Drive if Not Present
# ----------------------------
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1f-Cb2T3EcYCMW6zo5lhGD9eWwYN7spVr"
if not os.path.exists("similarity.pkl"):
    st.info("Downloading similarity.pkl from Google Drive...")
    r = requests.get(GDRIVE_URL, allow_redirects=True)
    with open("similarity.pkl", "wb") as f:
        f.write(r.content)
    st.success("Download complete!")

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Movie Suggester",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Custom CSS for a More Engaging UI
# ----------------------------
def local_css():
    st.markdown("""
        <style>
        /* ... your existing CSS code ... */
        </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Data Loading and Processing (Cached)
# ----------------------------
@st.cache_data
def load_movie_data():
    try:
        return pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'movie_list.pkl' is present.")
        return None

@st.cache_data
def load_similarity_matrix():
    try:
        return pickle.load(open('similarity.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Similarity matrix file not found.")
        return None

movies = load_movie_data()
similarity = load_similarity_matrix()

# ----------------------------
# API Helper Functions
# ----------------------------
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # fallback key

@st.cache_data
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = f"https://image.tmdb.org/t/p/w500/{data.get('poster_path')}" if data.get('poster_path') else "https://placehold.co/500x750/111827/FFFFFF?text=No+Poster"
        genres = [genre['name'] for genre in data.get('genres', [])]
        return {
            "poster": poster_path,
            "overview": data.get('overview', 'No overview available.'),
            "genres": genres,
            "rating": data.get('vote_average', 0)
        }
    except requests.exceptions.RequestException:
        return None

def recommend(movie):
    if movies is None or similarity is None or movie not in movies['title'].values: return []
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        movie_df = movies.iloc[i[0]]
        details = fetch_movie_details(movie_df.movie_id)
        recommended_movies.append({
            "id": movie_df.movie_id,
            "title": movie_df.title,
            "poster": details['poster'] if details else "https://placehold.co/500x750/111827/FFFFFF?text=Error"
        })
    return recommended_movies

# ----------------------------
# UI Rendering
# ----------------------------
local_css()
st.markdown('<p class="gradient-text">Movie Suggester</p>', unsafe_allow_html=True)
st.write("Discover your next favorite movie. Select one you like, and we'll find similar ones for you.")
st.markdown("---")

if movies is not None:
    st.markdown('<p class="select-label">Start by selecting a movie from the list below:</p>', unsafe_allow_html=True)
    selected_movie_name = st.selectbox('Search for a movie', movies['title'].values, label_visibility="collapsed")

    if selected_movie_name:
        st.subheader("🎬 Preview")
        selected_movie_id = movies[movies['title'] == selected_movie_name].iloc[0].movie_id
        details = fetch_movie_details(selected_movie_id)
        
        if details:
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.image(details['poster'])
            with col2:
                st.markdown(f"#### {selected_movie_name}")
                st.markdown(f"**⭐ Rating:** {details['rating']:.1f}/10")
                st.markdown(f"**🎭 Genres:** {', '.join(details['genres'])}")
                st.markdown("---")
                st.markdown(details['overview'])
        else:
            st.warning("Could not fetch details for this movie.")
    
    st.markdown("---")
    if st.button('✨ Find Similar Movies'):
        with st.spinner('Analyzing cosmic movie data...'):
            time.sleep(1)
            recommendations = recommend(selected_movie_name)
            
            if not recommendations:
                st.error("Could not find recommendations. Please try another movie.")
            else:
                st.subheader("🍿 You Might Also Like...")
                cols = st.columns(5, gap="large")
                for i, movie in enumerate(recommendations):
                    with cols[i]:
                        tmdb_url = f"https://www.themoviedb.org/movie/{movie['id']}"
                        st.markdown(f"""
                            <div class="movie-card">
                                <a href="{tmdb_url}" target="_blank">
                                    <img src="{movie['poster']}" class="movie-poster">
                                </a>
                                <a href="{tmdb_url}" target="_blank" class="movie-title-link">
                                    {movie['title']}
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
