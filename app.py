import streamlit as st

st.set_page_config(
    page_title="ABSA System",
    layout="wide"
)

st.title("Aspect-Based Sentiment Analysis (ABSA)")

st.markdown("""
## Selamat Datang 👋

Sistem ini digunakan untuk melakukan **Aspect-Based Sentiment Analysis (ABSA)**
pada ulasan produk e-commerce berbahasa Indonesia.

### Fitur Utama
- Ekstraksi aspek secara otomatis
- Identifikasi sentimen pada setiap aspek
- Implementasi model terbaik BiLSTM-CRF
- Eksplorasi dataset hasil anotasi

### Halaman Tersedia
- **Dataset Explorer**
- **Prediction**

Gunakan menu di sidebar untuk berpindah halaman.
""")
