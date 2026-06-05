from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
import pandas as pd

# ==========================================
# GROQ API KEY
# ==========================================

client = Groq(
    api_key=api_key
)
import os

api_key = os.getenv("GROQ_API_KEY")

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading papers...")

df = pd.read_csv(
    "data/clean_multimodal_papers.csv"
)

# ==========================================
# LOAD FAISS INDEX
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(
    "data/papers.index"
)

# ==========================================
# QUERY LOOP
# ==========================================

while True:

    question = input(
        "\nAsk a research question (type exit): "
    )

    if question.lower() == "exit":
        break

    # ======================================
    # EMBED QUERY
    # ======================================

    query_embedding = model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # ======================================
    # SEARCH FAISS
    # ======================================

    distances, indices = index.search(
        query_embedding,
        5
    )

    context = ""

    print("\n" + "=" * 80)
    print("SOURCES USED")
    print("=" * 80)

    for rank, idx in enumerate(indices[0], 1):

        paper = df.iloc[idx]

        title = str(
            paper.get("title", "No Title")
        )

        print(f"\n{rank}. {title}")

        context += "\n\nTITLE:\n"
        context += title

        if "abstract" in df.columns:

            context += "\n\nABSTRACT:\n"
            context += str(
                paper["abstract"]
            )

    # ======================================
    # BUILD PROMPT
    # ======================================

    prompt = f"""
You are an expert research assistant.

Use ONLY the information from the papers below.

{context}

Question:
{question}

Provide:
1. A detailed answer
2. Key trends
3. Important findings
4. A short conclusion
"""

    # ======================================
    # CALL LLM
    # ======================================

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # ======================================
    # DISPLAY ANSWER
    # ======================================

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        response.choices[0].message.content
    )

    print("\n" + "=" * 80)