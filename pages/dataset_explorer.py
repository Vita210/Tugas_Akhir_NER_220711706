import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("Dataset Explorer")

st.markdown("""
Dataset ulasan e-commerce yang sudah dianotasi menggunakan skema BILOU.
""")

# dummy / atau ganti load dataset kamu
try:
    df = st.session_state.df
except:
    df = pd.DataFrame({
        "text": ["produk bagus", "pengiriman lambat", "packing aman"],
        "label": ["positive", "negative", "neutral"]
    })

st.subheader("Sample Data")
st.dataframe(df)

st.subheader("Statistik")
st.write(f"Total data: {len(df)}")

st.subheader("Distribusi Label")

if "label" in df.columns:
    st.bar_chart(df["label"].value_counts())
