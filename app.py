import streamlit as st
from inference import predict 

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

    result = predict(text)

    tokens = result["tokens"]
    labels = result["labels"]
    extracted = result["extracted"]

    if not extracted:
        st.warning("Tidak ada aspek terdeteksi.")
        return

    st.subheader("Hasil Analisis")

    # ==============================
    # RESULT CARD UI (FIXED LAYOUT)
    # ==============================
    for item in extracted:

        frasa = item["frasa"]
        aspek = item["aspek"]
        sentimen = item["sentimen"]

        # tentukan style
        if sentimen.lower() == "positive":
            box = st.success
        elif sentimen.lower() == "negative":
            box = st.error
        else:
            box = st.info

        # card container
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown("**Frasa**")
                st.write(frasa)

            with col2:
                st.markdown("**Aspek**")
                st.write(aspek)

            with col3:
                st.markdown("**Sentimen**")
                box(sentimen)

    # ======================================
    # TOKEN LEVEL OUTPUT
    # ======================================
    if show_detail:

        st.markdown("### Token-Level Output")

        with st.container(border=True):
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
