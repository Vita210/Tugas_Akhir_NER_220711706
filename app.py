import streamlit as st

st.set_page_config(page_title="ABSA System", layout="wide")

st.title("Aspect-Based Sentiment Analysis (ABSA)")

# =========================
# SIDEBAR MENU (BUTTON STYLE)
# =========================
st.sidebar.title("Menu")

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home"):
    st.session_state.page = "Home"

if st.sidebar.button("Dataset Explorer"):
    st.session_state.page = "Dataset"

if st.sidebar.button("Prediction"):
    st.session_state.page = "Prediction"

# =========================
# PAGE ROUTING
# =========================
page = st.session_state.page

# =========================
# HOME
# =========================
if page == "Home":
    st.markdown("""
    ## Selamat datang 👋

    Aplikasi ini digunakan untuk:
    - Ekstraksi aspek dari ulasan e-commerce
    - Analisis sentimen berbasis aspek
    - Model: BiLSTM-CRF
    """)

    st.info("Gunakan menu di sidebar untuk navigasi.")

# =========================
# DATASET
# =========================
elif page == "Dataset":
    import dataset_explorer

# =========================
# PREDICTION
# =========================
elif page == "Prediction":
    import prediction
