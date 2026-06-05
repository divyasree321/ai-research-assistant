import streamlit as st
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/clean_multimodal_papers.csv"
    )

# ==========================================
# LOAD FAISS INDEX
# ==========================================

@st.cache_resource
def load_faiss():
    return faiss.read_index(
        "data/papers.index"
    )

# ==========================================
# LOAD RESOURCES
# ==========================================

model = load_model()
df = load_data()
index = load_faiss()

# ==========================================
# GROQ CLIENT
# ==========================================

client = Groq(
    api_key=api_key
)
import os

api_key = os.getenv("GROQ_API_KEY")
# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📚 Research Assistant")

st.sidebar.write(
    f"Total Papers: {len(df)}"
)

st.sidebar.write(
    "Powered by:"
)

st.sidebar.markdown("""
- Sentence Transformers
- FAISS
- Llama 3 (Groq)
- Topic Drift Analysis
""")

st.sidebar.markdown("---")

try:
    st.sidebar.subheader(
        "📈 Topic Drift"
    )

    st.sidebar.image(
        "src/figures/topic_drift.png",
        use_container_width=True
    )

except:
    st.sidebar.warning(
        "Topic drift graph not found."
    )

# ==========================================
# MAIN PAGE
# ==========================================

st.title(
    "📚 AI Research Assistant"
)

st.write(
    "Semantic Search + RAG + Topic Drift Analysis"
)

# ==========================================
# SHOW GRAPH
# ==========================================

st.subheader(
    "📈 Topic Drift Analysis"
)

try:

    st.image(
        "src/figures/topic_drift.png",
        use_container_width=True
    )

except:

    st.warning(
        "Topic drift graph not found."
    )

# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_input(
    "Ask a research question"
)

# ==========================================
# SEARCH BUTTON
# ==========================================

if st.button("Search"):

    if question.strip() == "":
        st.warning(
            "Please enter a question."
        )
        st.stop()

    # ======================================
    # EMBED QUESTION
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

    st.subheader(
        "📄 Sources Used"
    )

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        paper = df.iloc[idx]

        title = paper["title"]

        st.markdown(
            f"**{rank}. {title}**"
        )

        context += (
            f"\nTITLE:\n{title}\n"
        )

        if "abstract" in df.columns:

            context += (
                "\nABSTRACT:\n"
                + str(
                    paper["abstract"]
                )
                + "\n"
            )

    # ======================================
    # PROMPT
    # ======================================

    prompt = f"""
You are an expert research assistant.

Use ONLY the papers below.

{context}

Question:
{question}

Provide:

1. Detailed Answer
2. Key Findings
3. Trends
4. Conclusion
"""

    # ======================================
    # LLM RESPONSE
    # ======================================

    with st.spinner(
        "Generating answer..."
    ):

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    # ======================================
    # DISPLAY ANSWER
    # ======================================

    st.subheader(
        "🤖 Answer"
    )

    st.write(answer)