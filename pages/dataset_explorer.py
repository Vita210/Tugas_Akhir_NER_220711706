import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Dataset Explorer", layout="wide")

BASE_DIR = os.getcwd()
DATA_FILES = ["data_train_bilou.jsonl", "data_val_bilou.jsonl", "data_test_bilou.jsonl"]
LABEL_PATH = os.path.join(BASE_DIR, "data", "label_list.json")

@st.cache_data
def load_data():
    df_list = []
    for file in DATA_FILES:
        path = os.path.join(BASE_DIR, "data", file)
        if os.path.exists(path):
            df_list.append(pd.read_json(path, lines=True))
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Cek label_list.json
    labels = []
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
    return df, labels

if 'df' not in st.session_state or 'labels' not in st.session_state:
    st.session_state.df, st.session_state.labels = load_data()

st.title("📊 Dataset Explorer")
df = st.session_state.df

# DEBUG: Tampilkan nama kolom jika error
st.sidebar.write("Kolom yang tersedia:", df.columns.tolist())

# Deteksi nama kolom secara otomatis
token_col = next((c for c in ['tokens', 'text', 'sentence'] if c in df.columns), None)
label_col = next((c for c in ['labels', 'tags', 'tag', 'ner_tags'] if c in df.columns), None)

if not token_col or not label_col:
    st.error(f"Gagal mendeteksi kolom! Kolom ditemukan: {df.columns.tolist()}. Pastikan data memiliki kolom untuk token dan label.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Total Sampel", len(df))
col2.metric("Jumlah Label Unik", len(st.session_state.labels))

st.divider()

search_query = st.text_input("Cari kata dalam teks ulasan:")
filtered_df = df
if search_query:
    filtered_df = df[df[token_col].apply(lambda x: search_query.lower() in " ".join(x).lower())]

n_rows = st.slider("Jumlah sampel yang ditampilkan:", 5, 50, 10)

for i, row in filtered_df.head(n_rows).iterrows():
    with st.expander(f"Ulasan #{i+1}: {' '.join(row[token_col][:10])}..."):
        st.write("**Teks Lengkap:**")
        st.text(" ".join(row[token_col]))
        
        st.write("**Anotasi BILOU:**")
        cols = st.columns(len(row[token_col]))
        for idx, (token, label) in enumerate(zip(row[token_col], row[label_col])):
            with cols[idx]:
                st.caption(token)
                st.code(label)

st.divider()
st.subheader("🏷️ Distribusi Label")
all_labels = [label for sublist in df[label_col] for label in sublist]
st.bar_chart(pd.Series(all_labels).value_counts())
