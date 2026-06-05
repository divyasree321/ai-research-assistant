
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load demo dataset
df = pd.read_csv("data/demo_papers.csv")

# Load FAISS index
index = faiss.read_index("data/papers.index")


def retrieve(query, top_k=20):
    emb = model.encode([query])
    emb = np.array(emb, dtype="float32")

    distances, indices = index.search(emb, top_k)

    papers = []

    for idx in indices[0]:
        # Skip invalid indices
        if idx < 0:
            continue

        # Prevent out-of-bounds errors
        if idx >= len(df):
            continue

        papers.append(df.iloc[idx].to_dict())

    # Fallback if FAISS returns no valid results
    if len(papers) == 0:
        papers = df.head(min(top_k, len(df))).to_dict("records")

    return papers
