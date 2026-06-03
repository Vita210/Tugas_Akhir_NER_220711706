import streamlit as st
import json
import pickle
import re
import os
import torch
import torch.nn as nn
from torchcrf import CRF

# ==========================================
# CONFIG
# ==========================================
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# LOAD FILES
# ==========================================
with open(os.path.join(ROOT_DIR, "vocab.pkl"), "rb") as f:
    vocab_w2i = pickle.load(f)

with open(os.path.join(ROOT_DIR, "tag2idx.pkl"), "rb") as f:
    tag2idx = pickle.load(f)

with open(os.path.join(ROOT_DIR, "idx2tag.pkl"), "rb") as f:
    idx2tag = pickle.load(f)

with open(
    os.path.join(ROOT_DIR, "metadata.json"),
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

best_params = metadata["best_params"]

pretrained_embeddings = torch.load(
    os.path.join(ROOT_DIR, "embedding_matrix.pt"),
    map_location=DEVICE
)

# ==========================================
# MODEL
# ==========================================
class BiLSTM_CRF_Model(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_tags,
        pretrained_embeddings
    ):
        super().__init__()

        embedding_dim = pretrained_embeddings.size(1)

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings,
            freeze=False,
            padding_idx=0
        )

        self.dropout = nn.Dropout(0.5)

        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_tags
        )

        self.crf = CRF(
            num_tags,
            batch_first=True
        )

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.bilstm(x)
        return self.fc(x)

    def decode(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(
            emissions,
            mask=mask
        )

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    model = BiLSTM_CRF_Model(
        vocab_size=len(vocab_w2i),
        hidden_dim=best_params["hidden_dim"],
        num_tags=len(tag2idx),
        pretrained_embeddings=pretrained_embeddings
    )

    model.load_state_dict(
        torch.load(
            os.path.join(ROOT_DIR, "bilstm_crf.pt"),
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# ==========================================
# PREPROCESS
# ==========================================
def preprocess_text(text):

    text = text.strip().lower()

    if not text:
        return []

    return re.findall(
        r"\w+|[^\w\s]",
        text,
        re.UNICODE
    )

# ==========================================
# PREDICT
# ==========================================
def predict(text):

    tokens = preprocess_text(text)

    if not tokens:
        return [], []

    encoded = [
        vocab_w2i.get(
            t,
            vocab_w2i["[UNK]"]
        )
        for t in tokens
    ]

    x = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(DEVICE)

    mask = (x != 0)

    with torch.no_grad():
        decoded = model.decode(
            x,
            mask
        )[0]

    labels = [
        idx2tag[i]
        for i in decoded
    ]

    return tokens, labels

# ==========================================
# EXTRACT ENTITY
# ==========================================
def split_label(label):

    parts = label.split("_")

    if len(parts) == 1:
        return parts[0], "neutral"

    return "_".join(parts[:-1]), parts[-1]

def extract(tokens, labels):

    results = []

    current_tokens = []
    current_entity = None

    for token, label in zip(tokens, labels):

        if label in ["O", "[PAD]"]:

            if current_tokens:

                aspek, sentimen = split_label(
                    current_entity
                )

                results.append({
                    "frasa": " ".join(current_tokens),
                    "aspek": aspek,
                    "sentimen": sentimen
                })

            current_tokens = []
            current_entity = None
            continue

        if "-" not in label:
            continue

        prefix, entity = label.split("-", 1)

        if prefix == "U":

            aspek, sentimen = split_label(entity)

            results.append({
                "frasa": token,
                "aspek": aspek,
                "sentimen": sentimen
            })

        elif prefix == "B":

            current_tokens = [token]
            current_entity = entity

        elif prefix in ["I", "L"]:

            current_tokens.append(token)

            if prefix == "L":

                aspek, sentimen = split_label(entity)

                results.append({
                    "frasa": " ".join(current_tokens),
                    "aspek": aspek,
                    "sentimen": sentimen
                })

                current_tokens = []
                current_entity = None

    return results

# ==========================================
# UI
# ==========================================
st.title("🔮 Prediction")

st.markdown("""
Masukkan ulasan produk e-commerce untuk
mendeteksi aspek dan sentimen menggunakan
model **BiLSTM-CRF**.
""")

text = st.text_area(
    "Input Review",
    height=150,
    placeholder="Contoh: packing aman tapi pengiriman lambat"
)

show_detail = st.checkbox(
    "Tampilkan token-level output"
)

def display_results(data):

    if not data:
        st.warning("Tidak ada aspek terdeteksi.")
        return

    st.subheader("Hasil Analisis")

    cols = st.columns(3)

    for i, item in enumerate(data):

        sentimen = item["sentimen"].lower()

        color_map = {
            "positive": "green",
            "negative": "red",
            "neutral": "blue"
        }

        color = color_map.get(
            sentimen,
            "gray"
        )

        with cols[i % 3].container(border=True):

            st.markdown(
                f"**Frasa:** `{item['frasa']}`"
            )

            st.markdown(
                f"**Aspek:** {item['aspek']}"
            )

            st.markdown(
                f"**Sentimen:** :{color}[{sentimen.upper()}]"
            )

if st.button("Analisis"):

    if not text.strip():
        st.warning("Masukkan teks terlebih dahulu.")

    else:

        with st.spinner("Memproses ulasan..."):

            tokens, labels = predict(text)

            extracted = extract(
                tokens,
                labels
            )

        display_results(extracted)

        if show_detail:

            st.divider()

            st.subheader(
                "Token-Level Output"
            )

            st.table([
                {
                    "Token": t,
                    "Label": l
                }
                for t, l in zip(
                    tokens,
                    labels
                )
            ])
