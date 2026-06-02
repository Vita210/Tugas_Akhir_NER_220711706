import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from utils.data_loader import load_data

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Dataset Explorer")

st.markdown("""
Halaman ini digunakan untuk mengeksplorasi dataset ulasan e-commerce yang digunakan
dalam penelitian Aspect-Based Sentiment Analysis (ABSA).
""")

# ==========================================
# LOAD DATA (SESSION STATE)
# ==========================================
if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# ==========================================
# DATA OVERVIEW
# ==========================================
st.subheader("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Data", df.shape[0])

with col2:
    st.metric("Jumlah Kolom", df.shape[1])

with col3:
    st.metric("Missing Value", df.isnull().sum().sum())

st.divider()

# ==========================================
# SAMPLE DATA
# ==========================================
st.subheader("🔍 Sample Data")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ==========================================
# LABEL DISTRIBUTION
# ==========================================
st.subheader("📊 Distribusi Label Sentimen")

sentiment_cols = ["fuel", "machine", "part"]

for col in sentiment_cols:
    st.markdown(f"### {col.capitalize()}")

    fig, ax = plt.subplots()

    counts = df[col].value_counts()

    ax.bar(counts.index, counts.values)

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Jumlah")
    ax.set_title(f"Distribusi {col}")

    st.pyplot(fig)

st.divider()

# ==========================================
# SAMPLE SENTENCES PER SENTIMENT
# ==========================================
st.subheader("📝 Contoh Kalimat Berdasarkan Sentimen")

selected_aspect = st.selectbox(
    "Pilih aspek:",
    sentiment_cols
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Negative")
    neg_samples = df[df[selected_aspect] == "negative"]["Sentence"].head(3)
    for sent in neg_samples:
        st.write(f"- {sent}")

with col2:
    st.markdown("### Neutral")
    neu_samples = df[df[selected_aspect] == "neutral"]["Sentence"].head(3)
    for sent in neu_samples:
        st.write(f"- {sent}")

with col3:
    st.markdown("### Positive")
    pos_samples = df[df[selected_aspect] == "positive"]["Sentence"].head(3)
    for sent in pos_samples:
        st.write(f"- {sent}")

st.divider()

# ==========================================
# SIMPLE STATISTICS
# ==========================================
st.subheader("📈 Statistik Tambahan")

sent_lengths = df["Sentence"].astype(str).apply(lambda x: len(x.split()))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rata-rata Panjang Kalimat", f"{sent_lengths.mean():.2f}")

with col2:
    st.metric("Min Panjang Kalimat", sent_lengths.min())

with col3:
    st.metric("Max Panjang Kalimat", sent_lengths.max())

with col4:
    st.metric("Median", sent_lengths.median())