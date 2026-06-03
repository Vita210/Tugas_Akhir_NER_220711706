import streamlit as st
import matplotlib.pyplot as plt
from utils.visualization import plot_label_distribution
from utils.data_loader import load_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")

if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Access data from session state
df = st.session_state.df

# Show basic dataset info
st.subheader("Dataset Overview")
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")

# Display sample data
st.subheader("Sample Data")
st.dataframe(df.head())

# Show label distribution
st.subheader("Label Distribution")
cols = st.columns(3)

for i, column in enumerate(['fuel', 'machine', 'part']):
    with cols[i % 3]:
        fig = plot_label_distribution(df, column)
        st.pyplot(fig)

# Sample sentences per sentiment
st.subheader("Sample Sentences by Sentiment")

# Select sentiment to explore
sentiment_to_explore = st.selectbox(
    "Choose sentiment to explore:",
    ["fuel", "machine", "part"]
)

# Display examples for each sentiment value
st.write(f"### {sentiment_to_explore.capitalize()} Sentiment Examples")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("#### Negative")
    for _, row in df[df[sentiment_to_explore] == 'negative'].head(3).iterrows():
        st.write(f"- {row['Sentence']}")

with col2:
    st.write("#### Neutral")
    for _, row in df[df[sentiment_to_explore] == 'neutral'].head(3).iterrows():
        st.write(f"- {row['Sentence']}")

with col3:
    st.write("#### Positive")
    for _, row in df[df[sentiment_to_explore] == 'positive'].head(3).iterrows():
        st.write(f"- {row['Sentence']}")
