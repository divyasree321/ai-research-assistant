import pandas as pd
import re

print("Loading papers...")

# Load dataset
df = pd.read_csv("data/multimodal_papers.csv")

# Remove rows with missing abstracts
df = df.dropna(subset=["abstract"])

# Sample 50,000 papers for faster processing
df = df.sample(
    n=50000,
    random_state=42
)

print(f"Sampled {len(df)} papers")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

print("Cleaning abstracts...")

# Apply cleaning
df["clean_abstract"] = df["abstract"].apply(clean_text)

# Save cleaned dataset
output_file = "data/clean_multimodal_papers.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n========================")
print("Done!")
print("Saved to:", output_file)
print("Dataset shape:", df.shape)
print("========================")