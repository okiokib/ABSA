import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import Word2Vec

# Pastikan nltk sudah diunduh
nltk.download('punkt')

# Fungsi untuk memuat data
@st.cache(allow_output_mutation=True)
def load_data(path):
    data = pd.read_csv(path)  # Membaca file CSV
    return data

# Fungsi untuk memuat model Word2Vec yang telah dilatih
@st.cache(allow_output_mutation=True)
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# Fungsi untuk memuat dan melatih model SVM
@st.cache(allow_output_mutation=True)
def load_and_train_models(data, word2vec_model):
    def vectorize_text(text):
        words = word_tokenize(text)
        vector = sum([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], [])
        return vector

    X = data['Ulasan'].apply(vectorize_text).tolist()
    X = pd.DataFrame(X).fillna(0).values  # Mengubah ke DataFrame dan mengisi nilai NaN

    svm_sentiment = SVC(kernel='linear')
    svm_sentiment.fit(X, data['sentimen'])

    svm_aspect = SVC(kernel='linear')
    svm_aspect.fit(X, data['Aspect'])

    return svm_sentiment, svm_aspect

# Fungsi untuk memproses dan menganalisis ulasan
def analyze_review(review, svm_sentiment, svm_aspect, word2vec_model):
    sentences = sent_tokenize(review)
    results = []
    aspect_labels = {0: 'fasilitas', 1: 'Staf/Layanan', 2: 'kebersihan', 3: 'lokasi', 4: 'Other'}

    for sentence in sentences:
        words = word_tokenize(sentence)
        sentence_vector = sum([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], [])
        sentence_vector = pd.DataFrame([sentence_vector]).fillna(0).values  # Mengubah ke DataFrame dan mengisi nilai NaN

        predicted_sentiment = svm_sentiment.predict(sentence_vector)
        predicted_aspect = svm_aspect.predict(sentence_vector)
        aspect = aspect_labels.get(predicted_aspect[0], 'Other')
        sentiment = 'positif' if predicted_sentiment[0] == 1 else 'negatif'
        results.append(f"aspek: {aspect} sentimen: {sentiment} - {sentence}")

    return results

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("Analisis Sentimen dan Aspek Ulasan dengan Word2Vec")
    st.sidebar.title("Menu")

    menu = st.sidebar.radio("Pilih menu:", ["Ulasan Baru"])

    if menu == "Ulasan Baru":
        st.subheader("Analisis Ulasan Baru")
        
        # Muat data dan model Word2Vec
        data = load_data('coba.csv')  # Ubah ke path yang sesuai
        word2vec_model = load_word2vec_model('word2vec.model')  # Ubah ke path model Word2Vec Anda

        if data is not None and word2vec_model is not None:
            st.write("Kolom dalam dataset:", data.columns)
            svm_sentiment, svm_aspect = load_and_train_models(data, word2vec_model)
        
            if svm_sentiment and svm_aspect:
                # Input ulasan
                review = st.text_area("Masukkan ulasan baru:")

                if st.button("Analyze"):
                    if review:
                        # Analisis ulasan
                        results = analyze_review(review, svm_sentiment, svm_aspect, word2vec_model)

                        # Tampilkan hasil
                        for result in results:
                            st.write(result)
                    else:
                        st.warning("Masukkan ulasan terlebih dahulu.")
        else:
            st.error("Gagal memuat data atau model Word2Vec. Periksa file CSV dan model Word2Vec Anda.")

if __name__ == "__main__":
    main()
