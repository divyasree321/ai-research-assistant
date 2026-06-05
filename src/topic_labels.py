import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

print("Loading data...")

df = pd.read_csv("data/topic_assignments.csv")

for topic in sorted(df["topic"].unique()):

    topic_docs = df[df["topic"] == topic]

    if len(topic_docs) < 10:
        continue

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=20
    )

    X = vectorizer.fit_transform(
        topic_docs["clean_abstract"]
    )

    words = vectorizer.get_feature_names_out()

    print("\n" + "="*50)
    print(f"TOPIC {topic}")
    print("="*50)
    print(", ".join(words))