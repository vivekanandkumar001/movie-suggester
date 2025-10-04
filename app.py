import streamlit as st
import pickle
import pandas as pd
import requests
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="🎬 Movie Suggester",
    layout="wide",
    page_icon="🎥"
)

st.title("🎥 Movie Suggester App")
st.write("Get movie recommendations instantly using AI-based similarity search!")

# ✅ Direct-download Google Drive links (convert your share links)
MOVIE_LIST_URL = "https://drive.google.com/uc?export=download&id=1H2mg7XgjOsffKgHhRWgper6bEEIAv03F"
SIMILARITY_URL = "https://drive.google.com/uc?export=download&id=1f-Cb2T3EcYCMW6zo5lhGD9eWwYN7spVr"

@st.cache_data
def load_data():
    try:
        movie_list_response = requests.get(MOVIE_LIST_URL)
        movie_list_response.raise_for_status()
        movies_dict = pickle.load(io.BytesIO(movie_list_response.content))
        movies = pd.DataFrame(movies_dict)

        similarity_response = requests.get(SIMILARITY_URL)
        similarity_response.raise_for_status()
        similarity = pickle.load(io.BytesIO(similarity_response.content))

        return movies, similarity
    except Exception as e:
        st.error(f"⚠️ Failed to load data: {e}")
        st.stop()

movies, similarity = load_data()

def recommend(movie):
    if movie not in movies['title'].values:
        st.warning("Movie not found in the dataset.")
        return []
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

selected_movie_name = st.selectbox("Search or select a movie", movies['title'].values)

if st.button("Show Recommendation"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie_name)
        if recommendations:
            st.success("✅ Recommended Movies:")
            for movie in recommendations:
                st.write(f"- 🎬 {movie}")
