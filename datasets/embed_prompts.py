# pip install sentence-transformers pandas tqdm numpy

from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

# ----------------------
# CONFIG
# ----------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight, fast, good quality
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # auto GPU if available

# ----------------------
# LOAD DATA
# ----------------------
df = pd.read_json("data_mc.json", orient="records", lines=True)
questions = df["Question"].tolist()

# ----------------------
# LOAD MODEL
# ----------------------
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# ----------------------
# GENERATE EMBEDDINGS
# ----------------------
embeddings = []
for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Embedding prompts"):
    batch = questions[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(
        batch,
        show_progress_bar=False,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True  # directly get numpy arrays
    )
    embeddings.extend(batch_embeddings)

# ----------------------
# SAVE EMBEDDINGS
# ----------------------
embeddings = np.array(embeddings, dtype=np.float32)
np.save("question_embeddings.npy", embeddings)

