# WILL Project RAG

## Group ID

**Group 80**

---

## Team Members

- **Alexandro Theodore Gedowolo — s4078157**  
  Project Lead & RAG Architecture / Integration Developer

- **Venkata Naga Sai Animish Kocharlakota — s4204426**  
  Evaluation Lead

- **Abhishek Sharma — s4023078**  
  Research & Data Lead

---

## Project Topic

**A RAG-based research assistant that helps non-expert users understand scientific research about attention span and human cognitive performance.**

---

## Project Aim

The aim of this project is to develop a **Retrieval-Augmented Generation (RAG) system** that helps non-expert users understand scientific research related to **attention span and human cognitive performance**.

The system will retrieve relevant information from a curated collection of research papers and use an LLM to explain the findings in simple language while providing supporting sources.

---

## Target Users

The main target users are **general or non-expert users** who want to understand scientific research without having to read long and technical research papers.

---

## Planned RAG Architecture

```text
Research Papers
→ Text Extraction
→ Cleaning
→ Chunking + Metadata
→ DPR Context Encoder
→ Passage Embeddings
→ FAISS Index
→ User Question
→ DPR Question Encoder
→ Top-5 Retrieved Chunks
→ LLM Generator
→ Answer + Supporting Sources
````

---

## Current Technical Plan

* **Knowledge Base:** 5–8 research papers
* **Retrieval:** DPR
* **Vector Index:** FAISS
* **Retrieved Context:** Top-5 chunks
* **LLM Generator:** TBD
* **Evaluation:** Retrieval metrics + LLM-based answer evaluation
* **Prototype:** Simple Streamlit interface
* **Source Traceability:** Paper, page, and DOI/source information

---

## Research Focus

The project will focus on how **environmental and behavioural factors** may affect:

* Attention
* Cognitive performance
* Working memory
* Focus

Possible factors may include:

* Sleep
* Caffeine
* Light exposure
* Other relevant behavioural or environmental factors

---

## Repository Structure

```text
WILL-Project-RAG/
│
├── data/
├── docs/
├── prototype/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
│
├── target/
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Team Workflow

Each member has a primary lead area, but major components will be **reviewed and tested collaboratively**.

```text
Lead
→ Implement
→ Team Member Review / Test
→ Fix
→ Merge to Main
```
The `main` branch will be kept as the stable working version of the project


