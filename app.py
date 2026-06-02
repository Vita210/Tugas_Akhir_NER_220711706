import streamlit as st

st.set_page_config(page_title="ABSA System", layout="wide")

st.title("Aspect-Based Sentiment Analysis (ABSA)")
st.markdown("""
## Selamat datang 👋

Aplikasi ini digunakan untuk:
- Ekstraksi aspek dari ulasan e-commerce
- Analisis sentimen berbasis aspek
- Model: BiLSTM-CRF

### Menu:
- Dataset Explorer → eksplorasi dataset
- Prediction → uji model dengan input teks
""")

st.divider()

st.info("Gunakan sidebar untuk navigasi ke halaman lain.")
