from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading papers...")
df = pd.read_csv("data/clean_multimodal_papers.csv")

print("Loading FAISS index...")
index = faiss.read_index("data/papers.index")

while True:

    query = input("\nAsk a question (or type exit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        5
    )

    print("\n" + "=" * 80)
    print("TOP RESULTS")
    print("=" * 80)

    for rank, idx in enumerate(indices[0], 1):

        paper = df.iloc[idx]

        print(f"\nResult #{rank}")

        if "title" in df.columns:
            print("\nTitle:")
            print(paper["title"])

        if "abstract" in df.columns:
            print("\nAbstract:")
            print(str(paper["abstract"]))

        if "published" in df.columns:
            print("\nPublished:")
            print(paper["published"])

        print("\n" + "-" * 80)