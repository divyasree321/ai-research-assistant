import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

print("Loading cleaned dataset...")

df = pd.read_csv("data/clean_multimodal_papers.csv")

texts = df["clean_abstract"].tolist()

print(f"Total papers: {len(texts)}")

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)

# Save embeddings
np.save(
    "data/embeddings.npy",
    embeddings
)

print("\n====================")
print("Embeddings Saved!")
print("File: data/embeddings.npy")
print("====================")