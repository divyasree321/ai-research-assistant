# AI Research Assistant

An AI-powered Research Assistant for analyzing multimodal research papers, topic trends, and topic drift over time.

## Features

- Research paper retrieval
- Semantic search using embeddings
- Topic modeling and topic analysis
- Topic drift visualization
- AI-generated research summaries
- Research paper recommendations
- PDF export support
- Reranking for improved retrieval quality

---

## Project Structure

```
ai-research-assistant/
│
├── app.py
├── app_v2.py
├── requirements.txt
│
├── modules/
│   ├── llm.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── recommender.py
│   ├── database.py
│   └── pdf_export.py
│
├── src/
│   ├── preprocess.py
│   ├── search.py
│   ├── rag.py
│   ├── topic_model.py
│   ├── topic_labels.py
│   ├── drift_analysis.py
│   └── visualize.py
│
└── data/
    ├── demo_papers.csv
    └── demo_topic_assignments.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/divyasree321/ai-research-assistant.git
cd ai-research-assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Get a free API key from:

https://console.groq.com

---

## Run Application

```bash
streamlit run app.py
```

or

```bash
streamlit run app_v2.py
```

---

## Dataset

For deployment, a lightweight demo dataset is included:

- demo_papers.csv
- demo_topic_assignments.csv

Large datasets, embeddings, and indexes are excluded from GitHub to keep the repository lightweight.

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- FAISS
- Sentence Transformers
- BERTopic
- Groq API
- Plotly

---

## Future Improvements

- Real-time arXiv ingestion
- Advanced RAG pipeline
- Research trend forecasting
- Multi-agent research workflows
- Citation graph analysis

---

## Author

Divya Sree

GitHub:
https://github.com/divyasree321
