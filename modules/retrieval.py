import faiss
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

df = pd.read_csv(
    "data/clean_multimodal_papers.csv"
)

index = faiss.read_index(
    "data/papers.index"
)


def retrieve(query, top_k=20):

    emb = model.encode([query])

    emb = np.array(
        emb
    ).astype("float32")

    distances, indices = index.search(
        emb,
        top_k
    )

    papers = []

    for idx in indices[0]:

        papers.append(
            df.iloc[idx]
        )

    return papers