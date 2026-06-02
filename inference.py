import json
import pickle
import os
import torch
import torch.nn as nn
from torchcrf import CRF

# ==========================================
# CONFIG (FIXED FOR STREAMLIT DEPLOYMENT)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD VOCAB & LABELS
# ==========================================
with open(os.path.join(MODEL_DIR, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

with open(os.path.join(MODEL_DIR, "tag2idx.pkl"), "rb") as f:
    tag2idx = pickle.load(f)

with open(os.path.join(MODEL_DIR, "idx2tag.pkl"), "rb") as f:
    idx2tag = pickle.load(f)

with open(os.path.join(MODEL_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

best_params = metadata["best_params"]

# ==========================================
# LOAD EMBEDDING MATRIX
# ==========================================
pretrained_embeddings = torch.load(
    os.path.join(MODEL_DIR, "embedding_matrix.pt"),
    map_location=DEVICE
)

# ==========================================
# MODEL CLASS
# ==========================================
class BiLSTM_CRF_Model(nn.Module):

    def __init__(self, vocab_size, hidden_dim, num_tags, pretrained_embeddings):
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

        self.fc = nn.Linear(hidden_dim * 2, num_tags)

        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, mask):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.bilstm(embeds)
        emissions = self.fc(lstm_out)
        return emissions

    def decode(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask)

# ==========================================
# LOAD MODEL
# ==========================================
model = BiLSTM_CRF_Model(
    vocab_size=len(vocab.w2i),
    hidden_dim=best_params["hidden_dim"],
    num_tags=len(tag2idx),
    pretrained_embeddings=pretrained_embeddings
)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "bilstm_crf.pt"),
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("✓ BiLSTM-CRF loaded")

# ==========================================
# PREPROCESS
# ==========================================
def preprocess_text(text):
    text = text.strip()
    if not text:
        return []
    return text.split()

# ==========================================
# PREDICT LABELS
# ==========================================
def predict_labels(text):

    tokens = preprocess_text(text)

    if not tokens:
        return [], []

    encoded = [
        vocab.w2i.get(token.lower(), vocab.w2i["[UNK]"])
        for token in tokens
    ]

    x = torch.tensor([encoded], dtype=torch.long).to(DEVICE)

    mask = (x != 0)

    with torch.no_grad():
        decoded = model.decode(x, mask)[0]

    labels = [idx2tag[idx] for idx in decoded]

    return tokens, labels

# ==========================================
# SPLIT LABEL
# ==========================================
def split_label(label):
    parts = label.split("_")

    if len(parts) == 1:
        return parts[0], "neutral"

    aspek = "_".join(parts[:-1])
    sentimen = parts[-1]

    return aspek, sentimen

# ==========================================
# EXTRACT ABSA
# ==========================================
def extract_joint_absa(tokens, labels):

    results = []
    current_tokens = []
    current_entity = None

    for token, label in zip(tokens, labels):

        if label in ["O", "[PAD]"]:
            if current_tokens:
                aspek, sentimen = split_label(current_entity)
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

        # U
        if prefix == "U":
            aspek, sentimen = split_label(entity)
            results.append({
                "frasa": token,
                "aspek": aspek,
                "sentimen": sentimen
            })

        # B
        elif prefix == "B":

            if current_tokens:
                aspek, sentimen = split_label(current_entity)
                results.append({
                    "frasa": " ".join(current_tokens),
                    "aspek": aspek,
                    "sentimen": sentimen
                })

            current_tokens = [token]
            current_entity = entity

        # I / L
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

    # handle sisa entity
    if current_tokens:
        aspek, sentimen = split_label(current_entity)
        results.append({
            "frasa": " ".join(current_tokens),
            "aspek": aspek,
            "sentimen": sentimen
        })

    return results

# ==========================================
# MAIN PREDICT
# ==========================================
def predict(text):

    tokens, labels = predict_labels(text)

    extracted = extract_joint_absa(tokens, labels)

    return {
        "tokens": tokens,
        "labels": labels,
        "extracted": extracted
    }

# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":

    sample = "harga murah pengiriman cepat seller ramah"

    result = predict(sample)

    print("\nTOKENS")
    print(result["tokens"])

    print("\nLABELS")
    print(result["labels"])

    print("\nABSA")
    for item in result["extracted"]:
        print(item)
