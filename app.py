import streamlit as st

from inference import (
    predict_bilstm_crf,
    extract_joint_absa
)

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="ABSA E-Commerce",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================
st.title("Aspect-Based Sentiment Analysis (ABSA)")

st.markdown("""
Model yang digunakan:

- BiLSTM-CRF
- FastText Embedding
- Skema Label BILOU

Model akan mengekstraksi aspek dan sentimen
dari ulasan produk e-commerce.
""")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Pengaturan")

show_detail = st.sidebar.checkbox(
    "Tampilkan token-level output",
    value=False
)

# ==========================================
# INPUT
# ==========================================
text = st.text_area(
    "Masukkan ulasan:",
    height=150,
    placeholder="contoh: packing aman tapi pengiriman lambat"
)

# ==========================================
# SHOW RESULT
# ==========================================
def show_result(text):

    tokens, labels = predict_bilstm_crf(text)

    extracted = extract_joint_absa(tokens, labels)

    if not extracted:
        st.warning("Tidak ada aspek terdeteksi.")
        return

    st.subheader("Hasil Analisis")

    for item in extracted:

        frasa = item["frasa"]
        aspek = item["aspek"]
        sentimen = item["sentimen"]

        if sentimen.lower() == "positive":

            st.success(
                f"""
Frasa : {frasa}

Aspek : {aspek}

Sentimen : {sentimen}
"""
            )

        elif sentimen.lower() == "negative":

            st.error(
                f"""
Frasa : {frasa}

Aspek : {aspek}

Sentimen : {sentimen}
"""
            )

        else:

            st.info(
                f"""
Frasa : {frasa}

Aspek : {aspek}

Sentimen : {sentimen}
"""
            )

    # ======================================
    # TOKEN LEVEL
    # ======================================
    if show_detail:

        st.markdown("### Token-Level Output")

        for token, label in zip(tokens, labels):

            st.text(f"{token:20} -> {label}")


# ==========================================
# PROCESS
# ==========================================
if st.button("Analisis"):

    if not text.strip():

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        with st.spinner("Memproses ulasan..."):

            show_result(text)

        st.divider()

        st.caption(
            "Skema label menggunakan BILOU "
            "(Begin, Inside, Last, Unit, Outside)"
        )