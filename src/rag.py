from pathlib import Path
import sys


# Add the project root so project modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Project modules
from src.retrieval.search import (
    load_chunks,
    load_faiss_index,
    load_question_encoder,
    encode_question,
    retrieve_top_chunks,
)

from src.generation.generate_answer import generate_answer

TOP_K = 5

# Load the retrieval components once so they can be reused
def load_rag_components():
    print("Loading research chunks...")
    chunks = load_chunks()

    print("Loading FAISS index...")
    index = load_faiss_index()

    # The FAISS vectors must match the current chunk dataset
    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunks.jsonl do not match. "
            f"Index vectors: {index.ntotal}, "
            f"Chunks: {len(chunks)}"
        )

    print("Loading DPR question encoder...")
    tokenizer, model, device = load_question_encoder()

    return {
        "chunks": chunks,
        "index": index,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }

# Encode the question into the same vector space used by the indexed research passages
def retrieve_evidence(question, components):
    question_embedding = encode_question(
        question,
        components["tokenizer"],
        components["model"],
        components["device"],
    )

    # Retrieve the five most relevant chunks
    retrieved_chunks = retrieve_top_chunks(
        question_embedding,
        components["index"],
        components["chunks"],
        top_k=TOP_K,
    )

    return retrieved_chunks


def build_source_list(retrieved_chunks):
    # Create a compact source list for the final RAG output
    sources = []

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        sources.append(
            {
                "source_number": source_number,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "page": chunk["page"],
                "doi": chunk.get("doi") or "Not available",
                "score": chunk.get("score"),
            }
        )

    return sources

 # Load components automatically when run directly
def run_rag(question, components=None):
    # A prototype can pass preloaded components later to avoid reloading DPR for every question
    if components is None:
        components = load_rag_components()

    print("Retrieving evidence...")

    retrieved_chunks = retrieve_evidence(
        question,
        components,
    )

    print("Generating grounded answer...")

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    sources = build_source_list(
        retrieved_chunks
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }


def display_result(result):
    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print()
    print(result["answer"])

    print()
    print("=" * 80)
    print("SUPPORTING SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print()
        print(f"[Source {source['source_number']}]")
        print(f"Chunk ID: {source['chunk_id']}")
        print(f"Paper ID: {source['paper_id']}")
        print(f"Paper: {source['title']}")
        print(f"Page: {source['page']}")
        print(f"DOI: {source['doi']}")


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python src/rag.py '
            '"<question>"'
        )
        return

    question = " ".join(sys.argv[1:]).strip()

    result = run_rag(question)

    display_result(result)


if __name__ == "__main__":
    main()