import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from collections import Counter

st.title("📊 Dataset Explorer")

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
# DATASET OVERVIEW
# ==========================================
st.subheader("Dataset Overview")

st.write(f"Number of samples : {df.shape[0]}")
st.write(f"Number of features : {df.shape[1]}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Train", (df["split"] == "Train").sum())

with col2:
    st.metric("Validation", (df["split"] == "Validation").sum())

with col3:
    st.metric("Test", (df["split"] == "Test").sum())

# ==========================================
# SAMPLE DATA
# ==========================================
st.subheader("Sample Data")

preview_df = df.copy()

if "tokens" in preview_df.columns:

    preview_df["Review"] = preview_df["tokens"].apply(
        lambda x: " ".join(x)
        if isinstance(x, list)
        else str(x)
    )

    st.dataframe(
        preview_df[["Review", "split"]].head(10),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# LABEL DISTRIBUTION
# ==========================================
st.subheader("Label Distribution")

aspect_counter = Counter()
sentiment_counter = Counter()
prefix_counter = Counter()

for labels in df["labels"]:

    for label in labels:

        if label == "O":
            prefix_counter["O"] += 1
            continue

        prefix, entity = label.split("-", 1)

        prefix_counter[prefix] += 1

        if "_" in entity:

            aspect = "_".join(entity.split("_")[:-1])
            sentiment = entity.split("_")[-1]

            aspect_counter[aspect] += 1
            sentiment_counter[sentiment] += 1

# ==========================================
# 3 CHARTS
# ==========================================
cols = st.columns(3)

with cols[0]:

    st.markdown("#### Aspect Distribution")

    fig, ax = plt.subplots()

    ax.bar(
        aspect_counter.keys(),
        aspect_counter.values()
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

with cols[1]:

    st.markdown("#### Sentiment Distribution")

    fig, ax = plt.subplots()

    ax.bar(
        sentiment_counter.keys(),
        sentiment_counter.values()
    )

    st.pyplot(fig)

with cols[2]:

    st.markdown("#### BILOU Distribution")

    fig, ax = plt.subplots()

    ax.bar(
        prefix_counter.keys(),
        prefix_counter.values()
    )

    st.pyplot(fig)

# ==========================================
# SAMPLE REVIEWS BY ASPECT
# ==========================================
st.subheader("Sample Reviews by Aspect")

available_aspects = sorted(list(aspect_counter.keys()))

selected_aspect = st.selectbox(
    "Choose aspect:",
    available_aspects
)

examples = []

for _, row in df.iterrows():

    labels = row["labels"]

    found = False

    for lbl in labels:

        if selected_aspect in lbl:
            found = True
            break

    if found:

        if "tokens" in row:
            examples.append(
                " ".join(row["tokens"])
            )

    if len(examples) >= 9:
        break

col1, col2, col3 = st.columns(3)

with col1:

    st.write("#### Example 1-3")

    for text in examples[:3]:
        st.write(f"- {text}")

with col2:

    st.write("#### Example 4-6")

    for text in examples[3:6]:
        st.write(f"- {text}")

with col3:

    st.write("#### Example 7-9")

    for text in examples[6:9]:
        st.write(f"- {text}")

# ==========================================
# ANNOTATION EXAMPLES
# ==========================================
st.subheader("Annotation Examples")

for i in range(min(5, len(df))):

    row = df.iloc[i]

    with st.container(border=True):

        st.write(f"### Data {i+1}")

        st.write(
            " ".join(row["tokens"])
        )

        st.code(
            " | ".join(row["labels"])
        )

        st.caption(row["split"])
