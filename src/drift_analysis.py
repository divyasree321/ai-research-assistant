import pandas as pd

print("Loading topic assignments...")

df = pd.read_csv("data/topic_assignments.csv")

# Convert date column
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["year"] = df["date"].dt.year

# Remove rows with invalid dates
df = df.dropna(subset=["year"])

print("Years range:")
print(df["year"].min(), "->", df["year"].max())

# Topic counts per year
topic_trends = (
    df.groupby(["year", "topic"])
      .size()
      .reset_index(name="count")
)

topic_trends.to_csv(
    "src/output/topic_trends.csv",
    index=False
)

print("\nSaved:")
print("src/output/topic_trends.csv")

print("\nSample:")

print(topic_trends.head())