import pandas as pd

df = pd.read_csv("data/clean_multimodal_papers.csv")

df_small = df.sample(
    5000,
    random_state=42
)

df_small.to_csv(
    "data/demo_papers.csv",
    index=False
)

print("Created data/demo_papers.csv")
print("Number of papers:", len(df_small))