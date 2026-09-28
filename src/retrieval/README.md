## Retrieval

This directory handles the retrieval stage of the RAG system using DPR and FAISS

### `build_index.py`

Creates DPR passage embeddings from `chunks.jsonl` and builds the FAISS index

Run:

```bash
python src/retrieval/build_index.py
```

### `search.py`

Encodes a user question using the DPR Question Encoder and retrieves the Top-5 most relevant chunks.

Run:

```bash
python src/retrieval/search.py "your question here"
```

### Workflow

```text
chunks.jsonl
→ DPR Context Encoder
→ FAISS Index
→ DPR Question Encoder
→ Top-5 Chunks
```

If the dataset changes, run `chunk_text.py` and rebuild the FAISS index before using `search.py`
