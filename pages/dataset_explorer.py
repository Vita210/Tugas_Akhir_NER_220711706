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
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return df, labels

if 'df' not in st.session_state or 'labels' not in st.session_state:
    st.session_state.df, st.session_state.labels = load_data()

st.title("📊 Dataset Explorer")
df = st.session_state.df
TOKEN_COL = 'tokens'
LABEL_COL = 'bilou_tags'

st.subheader("🔍 Telusuri Data")
search_query = st.text_input("Cari kata dalam teks ulasan:")
filtered_df = df
if search_query:
    filtered_df = df[df[TOKEN_COL].apply(lambda x: search_query.lower() in " ".join(x).lower())]

n_rows = st.slider("Jumlah sampel:", 5, 50, 10)

for i, row in filtered_df.head(n_rows).iterrows():
    with st.expander(f"Ulasan #{i+1}: {' '.join(row[TOKEN_COL][:10])}..."):
        st.write("**Anotasi BILOU:**")
        
        # Menggunakan grid per baris agar tidak menyamping terus
        cols_per_row = 6 
        tokens = row[TOKEN_COL]
        labels = row[LABEL_COL]
        
        for j in range(0, len(tokens), cols_per_row):
            cols = st.columns(cols_per_row)
            for k in range(cols_per_row):
                if j + k < len(tokens):
                    with cols[k]:
                        st.caption(f"**{tokens[j+k]}**")
                        st.code(labels[j+k])

st.divider()
st.subheader("🏷️ Distribusi Label (Tanpa 'O')")
all_labels = [label for sublist in df[LABEL_COL] for label in sublist if label != 'O']
if all_labels:
    st.bar_chart(pd.Series(all_labels).value_counts())
