import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Dataset Explorer", layout="wide")

# ==========================================
# KONFIGURASI PATH
# ==========================================
BASE_DIR = os.getcwd()
DATA_FILES = ["data_train_bilou.jsonl", "data_val_bilou.jsonl", "data_test_bilou.jsonl"]
LABEL_PATH = os.path.join(BASE_DIR, "data", "label_list.json")
TOKEN_COL = 'tokens'
LABEL_COL = 'bilou_tags'

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df_list = []
    for file in DATA_FILES:
        path = os.path.join(BASE_DIR, "data", file)
        if os.path.exists(path):
            df_list.append(pd.read_json(path, lines=True))
    
    if not df_list:
        return pd.DataFrame(), []
        
    df = pd.concat(df_list, ignore_index=True)
    
    labels = []
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
            
    return df, labels

if 'df' not in st.session_state or 'labels' not in st.session_state:
    st.session_state.df, st.session_state.labels = load_data()

df = st.session_state.df

# ==========================================
# UI UTAMA
# ==========================================
st.title("📊 Dataset Explorer")

# Statistik Awal
col1, col2 = st.columns(2)
col1.metric("Total Sampel", len(df))
col2.metric("Jumlah Label Unik", len(st.session_state.labels))

st.divider()

# ==========================================
# TELUSURI DATA & FILTER
# ==========================================
st.subheader("🔍 Telusuri Data")

# 1. Filter Pencarian Teks
search_query = st.text_input("Cari kata dalam teks ulasan:")

# 2. Filter Berdasarkan Label (Fitur Tambahan)
all_entities = sorted(list(set([l for sublist in df[LABEL_COL] for l in sublist if l != 'O'])))
selected_labels = st.multiselect("Filter berdasarkan label entitas:", all_entities)

# Logika Filter
filtered_df = df
if search_query:
    filtered_df = filtered_df[filtered_df[TOKEN_COL].apply(lambda x: search_query.lower() in " ".join(x).lower())]
if selected_labels:
    filtered_df = filtered_df[filtered_df[LABEL_COL].apply(lambda x: any(l in selected_labels for l in x))]

n_rows = st.slider("Jumlah sampel yang ditampilkan:", 5, 50, 10)

for i, row in filtered_df.head(n_rows).iterrows():
    with st.expander(f"Ulasan #{i+1}: {' '.join(row[TOKEN_COL][:10])}..."):
        tokens = row[TOKEN_COL]
        labels = row[LABEL_COL]
        
        # Layout Vertikal (Sesuai keinginan Anda)
        for token, label in zip(tokens, labels):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.caption(f"**{token}**")
            with c2:
                st.text(label)

st.divider()

# ==========================================
# DISTRIBUSI LABEL
# ==========================================
st.subheader("🏷️ Distribusi Label (Tanpa 'O')")
all_labels = [label for sublist in df[LABEL_COL] for label in sublist if label != 'O']
if all_labels:
    st.bar_chart(pd.Series(all_labels).value_counts())
else:
    st.write("Tidak ada label entitas ditemukan.")
