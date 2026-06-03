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
    
    if not df_list:
        return pd.DataFrame(), []
    
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

# --- FUNGSI PEMBANTU: Ekstrak Kategori (Hapus prefix BILOU) ---
def get_base_label(label):
    if label == 'O': return 'O'
    # Mengambil bagian setelah '-' pertama, contoh: 'B-kualitas_positif' -> 'kualitas_positif'
    return label.split('-', 1)[1] if '-' in label else label

# --- FITUR ANALISIS ---
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("📏 Distribusi Panjang Kalimat")
    df['seq_len'] = df[TOKEN_COL].apply(len)
    st.bar_chart(df['seq_len'].value_counts().sort_index())

with col_b:
    st.subheader("🏷️ Filter Kategori Label")
    # Mengambil list kategori unik (tanpa prefix B/I/L/U)
    all_flat_labels = [get_base_label(l) for sublist in df[LABEL_COL] for l in sublist if l != 'O']
    unique_categories = sorted(list(set(all_flat_labels)))
    selected_categories = st.multiselect("Pilih kategori (tanpa tag BILOU):", unique_categories)

# --- TELUSURI DATA ---
st.divider()
st.subheader("🔍 Telusuri Data")

filtered_df = df
search_query = st.text_input("Cari kata dalam teks ulasan:")
if search_query:
    filtered_df = filtered_df[filtered_df[TOKEN_COL].apply(lambda x: search_query.lower() in " ".join(x).lower())]

if selected_categories:
    # Filter baris yang mengandung kategori yang dipilih
    mask = filtered_df[LABEL_COL].apply(
        lambda x: any(get_base_label(l) in selected_categories for l in x)
    )
    filtered_df = filtered_df[mask]

n_rows = st.slider("Jumlah sampel:", 5, 50, 10)

for i, row in filtered_df.head(n_rows).iterrows():
    with st.expander(f"Ulasan #{i+1}: {' '.join(row[TOKEN_COL][:10])}..."):
        tokens = row[TOKEN_COL]
        labels = row[LABEL_COL]
        
        for j in range(len(tokens)):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.caption(f"**{tokens[j]}**")
            with c2:
                st.text(labels[j])

# --- DISTRIBUSI LABEL ---
st.divider()
st.subheader("🏷️ Distribusi Label")
if all_flat_labels:
    st.bar_chart(pd.Series(all_flat_labels).value_counts())
