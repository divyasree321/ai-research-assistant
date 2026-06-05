import pandas as pd
import numpy as np

import umap
import hdbscan

print("Loading data...")

df = pd.read_csv(
    "data/clean_multimodal_papers.csv"
)

embeddings = np.load(
    "data/embeddings.npy"
)

print("Embeddings:", embeddings.shape)

print("\nRunning UMAP...")

umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

reduced_embeddings = umap_model.fit_transform(
    embeddings
)

print(
    "Reduced shape:",
    reduced_embeddings.shape
)

print("\nRunning HDBSCAN...")

cluster_model = hdbscan.HDBSCAN(
    min_cluster_size=100,
    metric="euclidean",
    cluster_selection_method="eom"
)

topics = cluster_model.fit_predict(
    reduced_embeddings
)

df["topic"] = topics

df.to_csv(
    "data/topic_assignments.csv",
    index=False
)

np.save(
    "data/reduced_embeddings.npy",
    reduced_embeddings
)

print("\n====================")
print("Topics discovered:",
      len(set(topics)))
print("Noise papers:",
      sum(topics == -1))
print("====================")