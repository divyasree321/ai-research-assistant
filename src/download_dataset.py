import kagglehub

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "Cornell-University/arxiv"
)

print("\nDataset downloaded to:")
print(path)