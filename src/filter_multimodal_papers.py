import json
import pandas as pd

INPUT_FILE = "data/arxiv-metadata-oai-snapshot.json"

KEYWORDS = [
    "clip",
    "vision transformer",
    "vit",
    "blip",
    "llava",
    "multimodal",
    "vision language",
    "image captioning",
    "visual question answering",
    "transformer",
    "image",
    "computer vision"
]

papers = []

print("Scanning dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        try:
            paper = json.loads(line)

            text = (
                paper.get("title", "") +
                " " +
                paper.get("abstract", "")
            ).lower()

            if any(keyword.lower() in text for keyword in KEYWORDS):

                papers.append({
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "authors": paper.get("authors"),
                    "categories": paper.get("categories"),
                    "date": paper.get("update_date")
                })

        except Exception:
            pass

        if i % 100000 == 0:
            print(f"Processed {i:,} papers")
            print(f"Matches found: {len(papers)}")

df = pd.DataFrame(papers)

df.drop_duplicates(
    subset=["title"],
    inplace=True
)

df.to_csv(
    "data/multimodal_papers.csv",
    index=False
)

print("\n==========================")
print("Finished!")
print("Papers found:", len(df))
print("==========================")