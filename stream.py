import streamlit as st
import pandas as pd
import pickle

@st.cache_data
def load_data():
    return pd.read_csv('data_cleaned.csv')

@st.cache_data
def load_model():
    with open('model_tfidf.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_data
def load_similarity_matrix():
    with open('cosine_similarity.pkl', 'rb') as file:
        return pickle.load(file)

data = load_data()
model = load_model()
similarity_matrix = load_similarity_matrix()

st.title("Movie and TV Show Recommendation System 🎬")
judul = st.text_input("Masukkan Judul")

if st.button("Cari Rekomendasi 🔍"):
    if judul not in data['title'].values:
        st.error("Judul tidak ditemukan")
    else:
        movie_idx = data[data['title'] == judul].index[0]
        
        similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, _ in similarity_scores[1:6]:
            title = data.iloc[idx]['title']
            type = data.iloc[idx]['type']
            genre = data.iloc[idx]['listed_in']
            recommendations.append({"Title": title, "Type": type, "Genre": genre})
            
        recommendations_df = pd.DataFrame(recommendations)
        
        st.subheader("5 Rekomendasi Konten 🎥")
        st.table(recommendations_df)
