## Prototype App

This folder contains the Streamlit prototype for the RAG system.

### `app.py`

The app provides a simple web interface where users can ask questions about the research papers.

The app sends the question to the RAG pipeline and displays:

- Detailed Answer
- Simple Summary
- Supporting Sources
- Paper title
- Page number
- DOI
- Retrieval score

### Requirements

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
````

Also make sure the `.env` file exists in the project root and contains:

```text
OPENAI_API_KEY=your_api_key_here
```

### Run the App

From the project root:

```bash
streamlit run prototype/app.py
```

The app will usually open automatically in your browser at:

```text
http://localhost:8501
```

### App Flow

```text
User Question
↓
Streamlit
↓
rag.py
↓
DPR + FAISS Retrieval
↓
Top-5 Evidence Chunks
↓
OpenAI API
↓
Detailed Answer + Simple Summary
↓
Supporting Sources
```

### Important

The FAISS index and `chunks.jsonl` must already be generated before running the app.

If the dataset changes, rebuild the chunks and FAISS index first.

