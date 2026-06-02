import streamlit as st
from inference import predict

st.set_page_config(page_title="Prediction", layout="wide")

st.title("🔮 Prediction - ABSA Model")

st.markdown("""
Masukkan ulasan produk untuk melihat **aspek dan sentimen** yang terdeteksi oleh model **BiLSTM-CRF**.
""")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.header("Informasi")
st.sidebar.info("""
Model: BiLSTM-CRF (Best Model)

Fungsi:
- Ekstraksi aspek (BILOU)
- Analisis sentimen per aspek
""")

# =========================
# INPUT
# =========================
text = st.text_area(
    "Input Review:",
    height=150,
    placeholder="contoh: packing aman tapi pengiriman lambat"
)

# =========================
# UI DISPLAY RESULTS
# =========================
def display_results(extracted_data):
    if not extracted_data:
        st.warning("Tidak ada aspek terdeteksi.")
        return

    st.subheader("📊 Hasil Analisis")

    cols = st.columns(3)

    color_map = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue"
    }

    for i, item in enumerate(extracted_data):
        frasa = item["frasa"]
        aspek = item["aspek"]
        sentimen = item["sentimen"].lower()

        color = color_map.get(sentimen, "gray")

        with cols[i % 3].container(border=True):
            st.markdown(f"**Frasa:** `{frasa}`")
            st.markdown(f"**Aspek:** {aspek}")
            st.markdown(f"**Sentimen:** :{color}[{sentimen.upper()}]")

# =========================
# TOKEN DISPLAY
# =========================
def show_tokens(tokens, labels):
    st.subheader("🧾 Token-Level Output")

    data = [{"Token": t, "Label": l} for t, l in zip(tokens, labels)]
    st.table(data)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Analisis"):

    if not text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Memproses analisis..."):
            result = predict(text)

        # hasil utama
        display_results(result["extracted"])

        st.divider()

        # token level
        show_tokens(result["tokens"], result["labels"])
