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

    train_df["split"] = "Train"
    val_df["split"] = "Validation"
    test_df["split"] = "Test"

    return pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )


df = load_dataset()

# ==========================================
# OVERVIEW
# ==========================================
st.subheader("📈 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Data", len(df))

with col2:
    st.metric("Jumlah Kolom", len(df.columns))

with col3:
    st.metric("Jumlah Split", df["split"].nunique())

st.divider()

# ==========================================
# DISTRIBUSI DATASET
# ==========================================
st.subheader("📊 Distribusi Dataset")

split_count = df["split"].value_counts()

col1, col2 = st.columns([2, 1])

with col1:
    st.bar_chart(split_count)

with col2:
    st.dataframe(
        split_count.reset_index().rename(
            columns={"index": "Split", "split": "Jumlah"}
        ),
        use_container_width=True
    )

st.divider()

# ==========================================
# SAMPLE DATA (RINGKAS)
# ==========================================
st.subheader("📝 Sample Data")

preview_df = df.copy()

# Jika ada kolom tokens
if "tokens" in preview_df.columns:
    preview_df["text"] = preview_df["tokens"].apply(
        lambda x: " ".join(x[:20]) if isinstance(x, list) else str(x)
    )

    display_cols = ["text", "split"]

elif "sentence" in preview_df.columns:
    display_cols = ["sentence", "split"]

else:
    display_cols = preview_df.columns.tolist()

st.dataframe(
    preview_df[display_cols].head(20),
    use_container_width=True
)

st.divider()

# ==========================================
# CONTOH ANOTASI
# ==========================================
st.subheader("🏷️ Contoh Anotasi BILOU")

sample_size = min(5, len(df))

for i in range(sample_size):

    row = df.iloc[i]

    with st.container(border=True):

        st.markdown(f"**Data #{i+1}**")

        if "tokens" in row:
            st.write("**Teks:**")
            st.write(" ".join(row["tokens"]))

        if "labels" in row:
            st.write("**Label BILOU:**")
            st.code(" | ".join(row["labels"]))

        st.caption(f"Split: {row['split']}")

st.divider()

# ==========================================
# INFORMASI DATASET
# ==========================================
st.subheader("ℹ️ Informasi Dataset")

train_count = (df["split"] == "Train").sum()
val_count = (df["split"] == "Validation").sum()
test_count = (df["split"] == "Test").sum()

st.write(f"""
- Data Train : **{train_count:,}**
- Data Validation : **{val_count:,}**
- Data Test : **{test_count:,}**
- Total Data : **{len(df):,}**
""")
