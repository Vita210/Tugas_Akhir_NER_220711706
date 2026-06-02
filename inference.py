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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD FILES
# ==========================================
with open(os.path.join(BASE_DIR, "vocab.pkl"), "rb") as f:
    vocab_w2i = pickle.load(f)

with open(os.path.join(BASE_DIR, "tag2idx.pkl"), "rb") as f:
    tag2idx = pickle.load(f)

with open(os.path.join(BASE_DIR, "idx2tag.pkl"), "rb") as f:
    idx2tag = pickle.load(f)

with open(os.path.join(BASE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

best_params = metadata["best_params"]

# ==========================================
# EMBEDDINGS
# ==========================================
pretrained_embeddings = torch.load(
    os.path.join(BASE_DIR, "embedding_matrix.pt"),
    map_location=DEVICE
)

# ==========================================
# MODEL
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
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.bilstm(x)
        emissions = self.fc(x)
        return emissions

    def decode(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask)


# ==========================================
# LOAD MODEL
# ==========================================
model = BiLSTM_CRF_Model(
    vocab_size=len(vocab_w2i),
    hidden_dim=best_params["hidden_dim"],
    num_tags=len(tag2idx),
    pretrained_embeddings=pretrained_embeddings
)

model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "bilstm_crf.pt"), map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

print("✓ Model loaded")

# ==========================================
# TEXT CLEANING (FIXED VERSION)
# ==========================================
def clean_token(token):
    token = str(token).strip()
    if not token:
        return None

    # hanya normalisasi ringan
    token = re.sub(r"\s+", " ", token)
    return token


def preprocess_text(text):
    text = text.strip().lower()
    if not text:
        return []

    # split sederhana tapi stabil untuk Indo e-commerce
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    cleaned = []
    for t in tokens:
        t = clean_token(t)
        if t:
            cleaned.append(t)

    return cleaned


# ==========================================
# PREDICT
# ==========================================
def predict_labels(text):
    tokens = preprocess_text(text)

    if not tokens:
        return [], []

    encoded = [
        vocab_w2i.get(t, vocab_w2i.get("[UNK]", 0))
        for t in tokens
    ]

    x = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    mask = (x != 0)

    with torch.no_grad():
        decoded = model.decode(x, mask)[0]

    labels = [idx2tag[i] for i in decoded]

    return tokens, labels


# ==========================================
# BILOU EXTRACTION
# ==========================================
def split_label(label):
    parts = label.split("_")
    if len(parts) == 1:
        return parts[0], "neutral"
    return "_".join(parts[:-1]), parts[-1]


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


def predict(text):
    tokens, labels = predict_labels(text)
    return {
        "tokens": tokens,
        "labels": labels,
        "extracted": extract_joint_absa(tokens, labels)
    }
