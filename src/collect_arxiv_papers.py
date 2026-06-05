import arxiv
import pandas as pd
import os
import time

# Create data folder
os.makedirs("data", exist_ok=True)

keywords = [
    "CLIP",
    "Vision Transformer",
    "ViT",
    "BLIP",
    "LLaVA",
    "Vision Language Model",
    "Multimodal AI",
    "Image Captioning",
    "Visual Question Answering"
]

papers = []

client = arxiv.Client()

for keyword in keywords:

    print(f"\nSearching: {keyword}")

    search = arxiv.Search(
        query=keyword,
        max_results=50,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    try:
        for result in client.results(search):

            papers.append({
                "title": result.title,
                "authors": ", ".join(
                    [author.name for author in result.authors]
                ),
                "published": result.published,
                "summary": result.summary.replace("\n", " "),
                "keyword": keyword,
                "pdf_url": result.pdf_url,
                "arxiv_url": result.entry_id
            })

        print(f"Finished: {keyword}")

        # Prevent HTTP 429 rate limit
        time.sleep(5)

    except Exception as e:
        print(f"Error for {keyword}: {e}")
        time.sleep(10)

# Create DataFrame
df = pd.DataFrame(papers)

# Remove duplicate papers
df.drop_duplicates(
    subset=["title"],
    inplace=True
)

# Save CSV
output_file = "data/multimodal_papers.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print("\n===================================")
print(f"Collected {len(df)} unique papers")
print(f"Saved to: {output_file}")
print("===================================")