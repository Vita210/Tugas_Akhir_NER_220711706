import streamlit as st
import pandas as pd
import os

st.title("📊 Dataset Explorer")

st.markdown("""
Halaman ini digunakan untuk mengeksplorasi dataset ulasan e-commerce
yang telah dianotasi menggunakan skema **BILOU**
(Begin, Inside, Last, Unit, Outside).
""")

# ==========================================
# LOAD DATASET
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(ROOT_DIR, "data", "data_train_bilou.jsonl")
val_path = os.path.join(ROOT_DIR, "data", "data_val_bilou.jsonl")
test_path = os.path.join(ROOT_DIR, "data", "data_test_bilou.jsonl")

@st.cache_data
def load_dataset():
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

df = load_dataset()

# ==========================================
# OVERVIEW
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Data", len(df))

with col2:
    st.metric("Jumlah Kolom", len(df.columns))

with col3:
    st.metric(
        "Jumlah Split",
        df["split"].nunique()
    )

st.divider()

# ==========================================
# SAMPLE DATA
# ==========================================
st.subheader("Sample Data")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.divider()

# ==========================================
# DISTRIBUSI SPLIT
# ==========================================
st.subheader("Distribusi Dataset")

split_count = df["split"].value_counts()

st.bar_chart(split_count)

st.write(split_count)
