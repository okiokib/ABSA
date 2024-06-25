import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import nltk
import streamlit as st
import pickle

# Pastikan nltk sudah diunduh
nltk.download('punkt')

# Load models and vectorizer
@st.cache(allow_output_mutation=True)
def load_models():
    with open('svm_sentiment.pkl', 'rb') as file:
        svm_sentiment = pickle.load(file)
    with open('svm_aspect.pkl', 'rb') as file:
        svm_aspect = pickle.load(file)
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return svm_sentiment, svm_aspect, vectorizer

svm_sentiment, svm_aspect, vectorizer = load_models()

# Load data
def load_data():
    path_file_excel = 'coba.xlsx'
    data = pd.read_excel(path_file_excel)
    return data

# Title
st.title("Ulasan Hotel Sentimen dan Aspek Analisis")

# Input
complex_review = st.text_area("Masukkan ulasan hotel:", "Kamar sangat nyaman dan fasilitasnya lengkap. Staff hotel sangat ramah dan membantu. Lokasinya strategis dekat dengan pusat perbelanjaan. Makanan di restoran enak dan bervariasi. Namun, kebersihan kamar mandi perlu ditingkatkan.")

if st.button("Analisis"):
    if complex_review:
        # Tokenize the complex review into sentences
        sentences = sent_tokenize(complex_review)

        # Analyze each sentence
        results = []
        for sentence in sentences:
            sentence_tfidf = vectorizer.transform([sentence])
            predicted_sentiment = svm_sentiment.predict(sentence_tfidf)
            predicted_aspect = svm_aspect.predict(sentence_tfidf)
            aspect_labels = {0: 'Fasilitas', 1: 'Staf/Layanan', 2: 'Kebersihan', 3: 'Lokasi', 4: 'Makanan'}
            aspect = aspect_labels.get(predicted_aspect[0], 'Unknown Aspect')
            sentiment = 'Positif' if predicted_sentiment[0] == 1 else 'Negatif'
            results.append(f"aspek:{aspect} sentimen:{sentiment} - {sentence}")

        # Display results
        st.write("### Hasil Analisis")
        for result in results:
            st.write(result)
    else:
        st.write("Silakan masukkan ulasan untuk dianalisis.")
