import streamlit as st

st.set_page_config(
    page_title="ABSA System",
    layout="wide"
)

st.title("Aspect-Based Sentiment Analysis (ABSA)")

st.markdown("""
## Selamat Datang 👋

Aplikasi ini digunakan untuk:

- Ekstraksi aspek dari ulasan e-commerce
- Analisis sentimen berbasis aspek
- Model terbaik: BiLSTM-CRF

Gunakan menu di sidebar untuk berpindah halaman.
""")
