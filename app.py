import streamlit as st
import pandas as pd
from inference import predict

st.set_page_config(page_title="ABSA System", layout="wide")

st.title("Aspect-Based Sentiment Analysis (ABSA)")

# =========================
# SIDEBAR MENU (NO DROPDOWN)
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Explorer", "Prediction"]
)

st.sidebar.info("Model: BiLSTM-CRF (Best Model)")

# =========================
# HOME
# =========================
if menu == "Home":
    st.subheader("📌 Deskripsi Sistem")

    st.write("""
    Sistem ini digunakan untuk melakukan **Aspect-Based Sentiment Analysis (ABSA)** pada ulasan e-commerce Bahasa Indonesia.

    Model:
    - CRF
    - BiGRU-CRF
    - BiLSTM-CRF (Best Model)

    Output:
    - Ekstraksi aspek
    - Sentimen tiap aspek
    """)

# =========================
# DATASET
# =========================
elif menu == "Dataset Explorer":

    st.subheader("📊 Dataset Explorer")

    try:
        df = st.session_state.df
    except:
        df = pd.DataFrame({
            "text": ["produk bagus", "pengiriman lambat", "packing aman"],
            "label": ["positive", "negative", "neutral"]
        })

    st.write("### Sample Data")
    st.dataframe(df)

    st.write("### Statistik")
    st.write(f"Jumlah data: {len(df)}")

# =========================
# PREDICTION
# =========================
elif menu == "Prediction":

    st.subheader("🔮 Prediction")

    text = st.text_area("Masukkan review:")

    if st.button("Analisis"):
        if not text.strip():
            st.warning("Input kosong")
        else:
            result = predict(text)

            st.write("### Tokens")
            st.write(result["tokens"])

            st.write("### Labels")
            st.write(result["labels"])

            st.write("### Hasil ABSA")
            st.write(result["extracted"])
