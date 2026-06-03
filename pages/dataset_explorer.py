import streamlit as st
import pandas as pd
import json
import os
import streamlit.components.v1 as components

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
    
    labels = []
    if os.path.exists(LABEL_PATH):
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

if search_query and TOKEN_COL in df.columns:
    filtered_df = df[df[TOKEN_COL].apply(lambda x: search_query.lower() in " ".join(x).lower())]

n_rows = st.slider("Jumlah sampel:", 5, 50, 10)

for i, row in filtered_df.head(n_rows).iterrows():
    with st.expander(f"Ulasan #{i+1}: {' '.join(row[TOKEN_COL][:10])}..."):
        st.write("**Anotasi BILOU:**")
        
        tokens = row[TOKEN_COL]
        labels = row[LABEL_COL]
        
        html_string = '<div style="display: flex; flex-wrap: wrap; gap: 8px; font-family: sans-serif;">'
        for token, label in zip(tokens, labels):
            bg_color = "#f0f2f6" if label == "O" else "#d1e7dd"
            text_color = "#333" if label == "O" else "#0f5132"
            border_color = "#ccc" if label == "O" else "#badbcc"
            
            html_string += f"""
            <div style="display: inline-block; padding: 5px 10px; background-color: {bg_color}; 
                        border-radius: 6px; border: 1px solid {border_color}; text-align: center;">
                <div style="font-weight: bold; font-size: 14px; color: #000; line-height: 1;">{token}</div>
                <div style="font-size: 10px; color: {text_color}; font-weight: bold; margin-top: 4px;">{label}</div>
            </div>
            """
        html_string += '</div>'
        
        # Menggunakan components.html agar dirender sebagai elemen visual
        components.html(html_string, height=150, scrolling=True)

st.divider()

st.subheader("🏷️ Distribusi Label (Tanpa 'O')")
if LABEL_COL in df.columns:
    all_labels = [label for sublist in df[LABEL_COL] for label in sublist if label != 'O']
    if all_labels:
        st.bar_chart(pd.Series(all_labels).value_counts())
    else:
        st.write("Tidak ada label entitas ditemukan.")
else:
    st.error("Kolom label tidak ditemukan.")
