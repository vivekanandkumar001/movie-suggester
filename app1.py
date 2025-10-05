import streamlit as st
import pickle
import pandas as pd
import requests
import time
import os
import gdown
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# Download Files if Missing
# ----------------------------
def download_file(drive_id, output_path):
    url = f"https://drive.google.com/uc?id={drive_id}"
    if not os.path.exists(output_path):
        try:
            st.info(f"📥 Downloading {output_path}...")
            gdown.download(url, output_path, quiet=False)
            if os.path.exists(output_path):
                st.success(f"✅ {output_path} downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download {output_path}: {e}")

# ----------------------------
# Google Drive direct download URLs
# ----------------------------
MOVIE_LIST_URL = "https://drive.google.com/uc?export=download&id=1H2mg7XgjOsffKgHhRWgper6bEEIAv03F"
SIMILARITY_URL = "https://drive.google.com/uc?export=download&id=1f-Cb2T3EcYCMW6zo5lhGD9eWwYN7spVr"


# Download files if not present
download_file(MOVIE_LIST_URL, "movie_list.pkl")
download_file(SIMILARITY_URL, "similarity.pkl")

# ----------------------------
# Custom CSS Styling
# ----------------------------
def local_css():
    st.markdown("""
        <style>
        @keyframes gradient { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
        .stApp { background: linear-gradient(-45deg, #111827, #1f2937, #374151, #4b5563); background-size: 400% 400%; animation: gradient 15s ease infinite; color: #F9FAFB; }
        .gradient-text { font-size: 3.5rem !important; font-weight: 800; background: -webkit-linear-gradient(45deg, #8B5CF6, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 1rem; }
        .select-label { color: #D1D5DB; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; }
        div[data-baseweb="select"] > div { background-color: rgba(31, 41, 55, 0.8); border: 1px solid #4B5563 !important; }
        div.stButton > button:first-child { border: none; border-radius: 9999px; padding: 0.75rem 1.5rem; background-color: #8B5CF6; color: #1F2937; font-weight: 700; transition: transform 0.2s ease, background-color 0.2s ease; }
        div.stButton > button:first-child:hover { transform: scale(1.05); background-color: #7C3AED; color: #111827; }
        .movie-card { background: rgba(31, 41, 55, 0.6); border-radius: 1rem; padding: 1rem; text-align: center; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .movie-card:hover { transform: translateY(-10px); box-shadow: 0 20px 25px -5px rgba(139,92,246,0.2); }
        .movie-poster { border-radius: 0.75rem; width: 100%; height: auto; margin-bottom: 0.75rem; }
        .movie-title-link { text-decoration: none; color: #D1D5DB; font-weight: 600; font-size: 1rem; }
        .movie-title-link:hover { color: #A78BFA; }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Load Movie Data & Similarity
# ----------------------------
@st.cache_data
def load_movie_data():
    try:
        with open('movie_list.pkl', 'rb') as f:
            return pd.DataFrame(pickle.load(f))
    except Exception as e:
        st.error(f"⚠️ Failed to load movie list: {e}")
        return None

@st.cache_data
def load_similarity_matrix():
    try:
        with open('similarity.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Failed to load similarity matrix: {e}")
        return None

movies = load_movie_data()
similarity = load_similarity_matrix()

# ----------------------------
# TMDB API
# ----------------------------
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except:
    TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

@st.cache_data
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        res = requests.get(url, timeout=10)
        data = res.json()
        poster = f"https://image.tmdb.org/t/p/w500/{data.get('poster_path')}" if data.get('poster_path') else "https://placehold.co/500x750/111827/FFFFFF?text=No+Poster"
        genres = [g['name'] for g in data.get('genres', [])]
        return {
            "poster": poster,
            "overview": data.get('overview', ''),
            "genres": genres,
            "rating": data.get('vote_average', 0)
        }
    except:
        return None

# ----------------------------
# Recommendation Logic
# ----------------------------
def recommend(movie_name):
    if movies is None or similarity is None or movie_name not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        m = movies.iloc[i[0]]
        details = fetch_movie_details(m.movie_id)
        recommendations.append({
            "id": m.movie_id,
            "title": m.title,
            "poster": details['poster'] if details else "https://placehold.co/500x750/111827/FFFFFF?text=Error"
        })
    return recommendations

# ----------------------------
# Streamlit UI
# ----------------------------
local_css()
st.markdown('<p class="gradient-text">Movie Suggester</p>', unsafe_allow_html=True)
st.write("Discover your next favorite movie 🎞️ — choose one you like, and we’ll find similar ones!")

st.markdown("---")

if movies is not None:
    st.markdown('<p class="select-label">🎬 Choose a Movie:</p>', unsafe_allow_html=True)
    selected_movie = st.selectbox("Select a Movie", movies['title'].values, label_visibility="collapsed")

    if selected_movie:
        selected_movie_id = movies[movies['title'] == selected_movie].iloc[0].movie_id
        details = fetch_movie_details(selected_movie_id)
        if details:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(details['poster'])
            with col2:
                st.markdown(f"### {selected_movie}")
                st.markdown(f"**⭐ Rating:** {details['rating']:.1f}/10")
                st.markdown(f"**🎭 Genres:** {', '.join(details['genres'])}")
                st.markdown("---")
                st.markdown(details['overview'])

    if st.button("✨ Find Similar Movies"):
        with st.spinner("Finding recommendations..."):
            time.sleep(1)
            recs = recommend(selected_movie)
            if recs:
                st.subheader("🍿 You Might Also Like")
                cols = st.columns(5)
                for i, r in enumerate(recs):
                    with cols[i]:
                        tmdb_url = f"https://www.themoviedb.org/movie/{r['id']}"
                        st.markdown(f"""
                            <div class="movie-card">
                                <a href="{tmdb_url}" target="_blank">
                                    <img src="{r['poster']}" class="movie-poster">
                                </a>
                                <a href="{tmdb_url}" target="_blank" class="movie-title-link">{r['title']}</a>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("⚠️ No recommendations found.")
else:
    st.error("⚠️ Failed to load movie data. Please recheck your Google Drive file links.")
