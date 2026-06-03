import streamlit as st
import pandas as pd

st.title("📊 Dataset Explorer")

st.markdown("""
Halaman ini digunakan untuk mengeksplorasi dataset ulasan e-commerce
yang telah dianotasi menggunakan skema **BILOU**
(Begin, Inside, Last, Unit, Outside).
""")

# jika nanti punya dataset asli
if "df" in st.session_state:
    df = st.session_state.df
else:
    df = pd.DataFrame({
        "text": [
            "produk bagus",
            "pengiriman lambat",
            "packing aman",
            "barang sesuai deskripsi",
            "harga terlalu mahal"
        ],
        "label": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative"
        ]
    })

col1, col2 = st.columns(2)

with col1:
    st.metric("Jumlah Data", len(df))

with col2:
    st.metric("Jumlah Kolom", len(df.columns))

st.divider()

st.subheader("Sample Data")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.divider()

if "label" in df.columns:

    st.subheader("Distribusi Label")

    label_count = df["label"].value_counts()

    st.bar_chart(label_count)
