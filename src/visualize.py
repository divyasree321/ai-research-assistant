import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "src/output/topic_trends.csv"
)

topics = df["topic"].unique()

plt.figure(figsize=(12,6))

for topic in topics:
    subset = df[df["topic"] == topic]

    plt.plot(
        subset["year"],
        subset["count"],
        label=f"Topic {topic}"
    )

plt.xlabel("Year")
plt.ylabel("Paper Count")
plt.title("Topic Drift Over Time")

plt.legend()

plt.tight_layout()

plt.savefig(
    "src/figures/topic_drift.png",
    dpi=300
)

print("\n GRAPH SAVED: src/figures/topic_drift.png")