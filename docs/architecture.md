# RAG ARCHITECTURE





### Project Goal



Build a research assistant that helps non-expert user understand research about environmental and behavioral factors affecting attention and cognitive performance



### Offline Pipeline

→ Text Extraction

→ Cleaning

→ Chunking

→ Metadata

→ DPR Context Encoder

→ Passage Embeddings

→ FAISS Index



### Online Pipeline

##### User Questions

→ DPR Question Encoder

→ Question Embedding

→ FAISS Search

→ Top-5 Chunks

→ Prompt Builder

→ LLM (TBD)

→ Answer + Supporting Sources



### Evaluations

##### Retrieval

* Recall@5
* NDCG@5

##### Generation

* Correctness
* Faithfulness
* Relevance
* Source Correctness
* Comprehensibility
* Insufficient Evidence behaviour



### Final Scope

* 5–8 research papers
* Dense retrieval
* FAISS
* Top-5 chunks
* One main LLM generator
* Answer with supporting sources
* 15–20 evaluation questions
* Simple Streamlit prototype
* No fine-tuning
* No multimodal RAG

