import streamlit as st
import pandas as pd
import os

from modules.retrieval import retrieve
from modules.reranker import rerank
from modules.llm import generate_answer

from modules.database import (
    init_db,
    save_search,
    get_recent_searches
)

from modules.analytics import (
    topic_counts
)

# ==========================================
# INIT
# ==========================================

init_db()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

try:
    df = pd.read_csv(
        "data/clean_multimodal_papers.csv"
    )
except:
    df = pd.DataFrame()

# ==========================================
# TITLE
# ==========================================

st.title("📚 AI Research Assistant")

st.markdown(
    """
Semantic Search + RAG + Topic Modeling + Topic Drift Analysis
"""
)

# ==========================================
# TOPIC DRIFT IMAGE
# ==========================================

st.subheader("📈 Topic Drift Analysis")

if os.path.exists(
    "src/figures/topic_drift.png"
):

    st.image(
        "src/figures/topic_drift.png",
        use_container_width=True
    )

else:

    st.warning(
        "Topic Drift graph not found."
    )

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "📊 Dashboard"
)

st.sidebar.metric(
    "Papers",
    len(df)
)

topics = topic_counts()

st.sidebar.metric(
    "Topics",
    len(topics)
)

# ==========================================
# TOPIC DISTRIBUTION
# ==========================================

with st.sidebar.expander(
    "📚 Topic Distribution",
    expanded=False
):

    if len(topics) > 0:

        topic_df = pd.DataFrame(
            {
                "Topic": list(
                    topics.keys()
                ),
                "Papers": list(
                    topics.values()
                )
            }
        )

        st.dataframe(
            topic_df,
            use_container_width=True
        )

        st.bar_chart(
            topic_df.set_index(
                "Topic"
            )
        )

    else:

        st.info(
            "No topic information available."
        )

# ==========================================
# RECENT SEARCHES
# ==========================================

st.sidebar.subheader(
    "🕒 Recent Searches"
)

recent = get_recent_searches()

if len(recent) == 0:

    st.sidebar.write(
        "No searches yet."
    )

else:

    for q in recent:

        st.sidebar.write(
            f"• {q}"
        )

# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_input(
    "Ask a research question"
)

# ==========================================
# SEARCH
# ==========================================

if st.button("Search"):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # ======================================
    # RETRIEVE
    # ======================================

    with st.spinner(
        "Searching papers..."
    ):

        papers = retrieve(
            question,
            top_k=20
        )

        papers = rerank(
            question,
            papers
        )

    st.sidebar.metric(
        "Retrieved",
        len(papers)
    )

    # ======================================
    # SHOW PAPERS
    # ======================================

    st.subheader(
        "📄 Retrieved Papers"
    )

    for i, paper in enumerate(
        papers,
        start=1
    ):

        title = paper.get(
            "title",
            "Unknown Title"
        )

        with st.expander(
            f"{i}. {title}"
        ):

            if "authors" in paper:

                st.write(
                    "**Authors:**",
                    paper["authors"]
                )

            if "date" in paper:

                st.write(
                    "**Date:**",
                    paper["date"]
                )

            st.write(
                "**Abstract:**"
            )

            st.write(
                str(
                    paper.get(
                        "abstract",
                        ""
                    )
                )[:1000]
            )

    # ======================================
    # GENERATE ANSWER
    # ======================================

    st.subheader(
        "🤖 AI Answer"
    )

    with st.spinner(
        "Generating answer..."
    ):

        try:

            answer = generate_answer(
                question,
                papers
            )

            save_search(
                question,
                answer
            )

            if (
                "Invalid API Key" in answer
                or
                "ERROR" in answer
                or
                "AuthenticationError" in answer
                or
                "AI Answer Unavailable" in answer
            ):

                st.warning(
                    answer
                )

            else:

                st.write(
                    answer
                )

                st.download_button(
                    label="📥 Download Research Report",
                    data=answer,
                    file_name="research_report.txt",
                    mime="text/plain"
                )

        except Exception as e:

            st.error(
                f"LLM Error: {str(e)}"
            )

            st.info(
                "Retrieved papers are still available above."
            )