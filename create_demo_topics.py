import pandas as pd

df = pd.read_csv("data/topic_assignments.csv")

df_small = df.sample(
    n=5000,
    random_state=42
)

df_small.to_csv(
    "data/demo_topic_assignments.csv",
    index=False
)

print("Created demo_topic_assignments.csv")