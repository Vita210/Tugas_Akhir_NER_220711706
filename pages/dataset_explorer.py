import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Dataset Explorer", layout="wide")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "data_all.jsonl")
LABEL_PATH = os.path.join(ROOT_DIR, "data", "label_list.json")

@st.cache_data
def load_data():
    df = pd.read_json(DATA_PATH, lines=True)
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return df, labels

if 'df' not in st.session_state or 'labels' not in st.session_state:
    st.session_state.df, st.session_state.labels = load_data()

st.title("📊 Dataset Explorer")
st.markdown("Eksplorasi ulasan e-commerce dan anotasi **BILOU**.")

df = st.session_state.df

col1, col2 = st.columns(2)
col1.metric("Total Sampel", len(df))
col2.metric("Jumlah Label Unik", len(st.session_state.labels))

st.divider()

st.subheader("🔍 Telusuri Data")

search_query = st.text_input("Cari kata dalam teks ulasan:")
filtered_df = df
if search_query:
    filtered_df = df[df['tokens'].apply(lambda x: search_query.lower() in " ".join(x).lower())]

n_rows = st.slider("Jumlah sampel yang ditampilkan:", 5, 50, 10)

st.write(f"Menampilkan {min(n_rows, len(filtered_df))} data:")

for i, row in filtered_df.head(n_rows).iterrows():
    with st.expander(f"Ulasan #{i+1}: {' '.join(row['tokens'][:10])}..."):
        st.write("**Teks Lengkap:**")
        st.text(" ".join(row['tokens']))
        
        st.write("**Anotasi BILOU:**")
        cols = st.columns(len(row['tokens']))
        for idx, (token, label) in enumerate(zip(row['tokens'], row['labels'])):
            with cols[idx]:
                st.caption(token)
                st.code(label)

st.divider()

st.subheader("🏷️ Distribusi Label")
all_labels = [label for sublist in df['labels'] for label in sublist]
label_counts = pd.Series(all_labels).value_counts()

st.bar_chart(label_counts)

if st.checkbox("Lihat detail daftar label"):
    st.write(st.session_state.labels)
