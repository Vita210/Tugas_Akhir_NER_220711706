import streamlit as st
from inference import predict 

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="ABSA E-Commerce", layout="wide")

# ==========================================
# UI HELPER FUNCTIONS
# ==========================================
def display_results(extracted_data: list):
    """Menampilkan hasil ekstraksi dalam format grid (columns)."""
    if not extracted_data:
        st.warning("Tidak ada aspek yang terdeteksi.")
        return

    st.subheader("Hasil Analisis")
    
    cols = st.columns(3)
    
    for i, item in enumerate(extracted_data):
        frasa = item["frasa"]
        aspek = item["aspek"]
        sentimen = item["sentimen"].lower()

        color_map = {"positive": "green", "negative": "red", "neutral": "blue"}
        status_color = color_map.get(sentimen, "gray")
        
        with cols[i % 3].container(border=True):
            st.markdown(f"**Frasa:** `{frasa}`")
            st.markdown(f"**Aspek:** {aspek}")
            st.markdown(f"**Sentimen:** :{status_color}[{sentimen.upper()}]")

def show_token_details(tokens: list, labels: list):
    """Menampilkan detail token level dalam bentuk tabel."""
    st.markdown("### Token-Level Output")
    data = [{"Token": t, "Label": l} for t, l in zip(tokens, labels)]
    st.table(data)

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("Aspect-Based Sentiment Analysis (ABSA)")
    st.markdown("Model: **BiLSTM-CRF** untuk ekstraksi aspek dan sentimen.")

    st.sidebar.header("Informasi")
    st.sidebar.info("Aplikasi ini menggunakan model BiLSTM-CRF untuk mendeteksi aspek dan sentimen dalam ulasan produk elektronik e-commerce dari Tokopedia dan Shopee.")

    text = st.text_area(
        "Masukkan ulasan:",
        height=150,
        placeholder="Contoh: packing aman tapi pengiriman lambat"
    )

    show_detail = st.checkbox("Tampilkan token-level output", value=False)

    if st.button("Analisis"):
        if not text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
            return

        with st.spinner("Memproses ulasan..."):
            result = predict(text)
            
            display_results(result["extracted"])

            if show_detail:
                show_token_details(result["tokens"], result["labels"])

        st.divider()
        st.caption("Skema label menggunakan BILOU (Begin, Inside, Last, Unit, Outside)")

if __name__ == "__main__":
    main()
