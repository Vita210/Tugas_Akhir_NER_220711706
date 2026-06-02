import streamlit as st

st.set_page_config(page_title="ABSA System", layout="wide")

st.title("Aspect-Based Sentiment Analysis (ABSA)")

st.sidebar.title("Menu")

if st.sidebar.button("Home"):
    st.session_state.page = "home"
if st.sidebar.button("Dataset Explorer"):
    st.session_state.page = "dataset"
if st.sidebar.button("Prediction"):
    st.session_state.page = "prediction"

# =========================
# HOME
# =========================
if menu == "Home":
    st.subheader("📌 Deskripsi Sistem")

    st.write("""
    Sistem ini digunakan untuk melakukan **Aspect-Based Sentiment Analysis (ABSA)** pada ulasan e-commerce Bahasa Indonesia.

    Model yang digunakan:
    - CRF
    - BiGRU-CRF
    - BiLSTM-CRF (Best Model)

    Output sistem:
    - Ekstraksi aspek (contoh: pengiriman, kualitas)
    - Sentimen tiap aspek (positive / negative / neutral)
    """)

    st.info("Gunakan menu di sidebar untuk eksplor dataset atau melakukan prediksi.")

# =========================
# DATASET EXPLORER
# =========================
elif menu == "Dataset Explorer":

    import pandas as pd

    st.subheader("📊 Dataset Explorer")

    st.write("""
    Dataset berisi ulasan e-commerce yang sudah dianotasi menggunakan skema BILOU.
    """)

    # contoh dummy kalau dataset belum di-load
    try:
        df = st.session_state.df
    except:
        df = pd.DataFrame({
            "text": ["produk bagus", "pengiriman lambat", "packing aman"],
            "label": ["positive", "negative", "neutral"]
        })

    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("### Statistik")
    st.write(f"Jumlah data: {len(df)}")

# =========================
# PREDICTION
# =========================
elif menu == "Prediction":

    from inference import predict

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
