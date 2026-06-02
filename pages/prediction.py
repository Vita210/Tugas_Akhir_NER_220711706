import streamlit as st
from inference import predict

st.set_page_config(page_title="Prediction", layout="wide")

st.title("Prediction - ABSA Model")

st.markdown("""
Masukkan ulasan produk untuk melihat aspek dan sentimen yang terdeteksi oleh model BiLSTM-CRF.
""")

text = st.text_area(
    "Input Review:",
    placeholder="contoh: packing aman tapi pengiriman lambat"
)

def display_results(extracted_data):
    if not extracted_data:
        st.warning("Tidak ada aspek terdeteksi.")
        return

    st.subheader("Hasil Prediksi")

    cols = st.columns(3)

    for i, item in enumerate(extracted_data):
        with cols[i % 3].container(border=True):
            st.write(f"**Frasa:** {item['frasa']}")
            st.write(f"**Aspek:** {item['aspek']}")
            st.write(f"**Sentimen:** {item['sentimen']}")

if st.button("Analisis"):
    if text.strip() == "":
        st.warning("Masukkan teks dulu.")
    else:
        with st.spinner("Memproses..."):
            result = predict(text)

        display_results(result["extracted"])

        st.divider()
        st.subheader("Token Level")
        st.table([
            {"Token": t, "Label": l}
            for t, l in zip(result["tokens"], result["labels"])
        ])
