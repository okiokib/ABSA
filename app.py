import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.tokenize import sent_tokenize
import nltk

# Pastikan nltk sudah diunduh
nltk.download('punkt')

# Fungsi untuk memuat data
@st.cache(allow_output_mutation=True)
def load_data(path):
    data = pd.read_csv(path)  # Membaca file CSV
    return data

# Fungsi untuk memuat dan melatih model serta vectorizer
@st.cache(allow_output_mutation=True)
def load_and_train_models(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['review'])

    svm_sentiment = SVC(kernel='linear')
    svm_sentiment.fit(X, data['sentiment'])

    svm_aspect = SVC(kernel='linear')
    svm_aspect.fit(X, data['aspect'])

    return svm_sentiment, svm_aspect, vectorizer

# Fungsi untuk memproses dan menganalisis ulasan
def analyze_review(review, svm_sentiment, svm_aspect, vectorizer):
    sentences = sent_tokenize(review)
    results = []
    aspect_labels = {0: 'fasilitas', 1: 'Staf/Layanan', 2: 'kebersihan', 3: 'lokasi', 4: 'Other'}

    for sentence in sentences:
        sentence_tfidf = vectorizer.transform([sentence])
        predicted_sentiment = svm_sentiment.predict(sentence_tfidf)
        predicted_aspect = svm_aspect.predict(sentence_tfidf)
        aspect = aspect_labels.get(predicted_aspect[0], 'Other')
        sentiment = 'positif' if predicted_sentiment[0] == 1 else 'negatif'
        results.append(f"aspek: {aspect} sentimen: {sentiment} - {sentence}")

    return results

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("Analisis Sentimen dan Aspek Ulasan")
    st.sidebar.title("Menu")

    menu = st.sidebar.radio("Pilih menu:", ["Ulasan Baru"])

    if menu == "Ulasan Baru":
        st.subheader("Analisis Ulasan Baru")
        
        # Muat data dan latih model
        data = load_data('coba.csv')  # Ubah ke path yang sesuai
        svm_sentiment, svm_aspect, vectorizer = load_and_train_models(data)

        # Input ulasan
        review = st.text_area("Masukkan ulasan baru:")

        if st.button("Analyze"):
            if review:
                # Analisis ulasan
                results = analyze_review(review, svm_sentiment, svm_aspect, vectorizer)

                # Tampilkan hasil
                for result in results:
                    st.write(result)
            else:
                st.warning("Masukkan ulasan terlebih dahulu.")

if __name__ == "__main__":
    main()
