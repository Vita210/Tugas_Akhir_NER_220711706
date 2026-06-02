import streamlit as st

st.set_page_config(page_title="ABSA System", layout="wide")

st.title("ABSA System")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Explorer", "Prediction"]
)

if menu == "Home":
    st.subheader("Deskripsi Sistem")
    st.write("""
    Sistem ABSA untuk analisis aspek dan sentimen ulasan e-commerce Indonesia.

    Model:
    - CRF
    - BiGRU-CRF
    - BiLSTM-CRF (Best)

    Output:
    - aspek
    - sentimen
    """)

elif menu == "Dataset Explorer":
    import dataset_explorer

elif menu == "Prediction":
    import prediction
