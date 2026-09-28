from pathlib import Path
import json
import sys

# Third-party libraries, our choice based on the WALERT 
import faiss
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chunking, Indexes Dir path
CHUNKS_FILE = PROJECT_ROOT / "data" / "metadata" / "chunks.jsonl"
INDEX_FILE = PROJECT_ROOT / "target" / "indexes" / "dpr_faiss.index"


# DPR question encoder configuration
QUESTION_ENCODER_MODEL = "facebook/dpr-question_encoder-multiset-base"

# Number of chunks returned by FAISS, return TOP 5 like WALLERT
TOP_K = 5

# DPR is based on BERT and supports up to 512 tokens
MAX_LENGTH = 512

# Load chunks in exactly the same order used when building
def load_chunks():
    # the FAISS index, FAISS result positions depend on this order
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks

# Load the previously generated FAISS passage index
def load_faiss_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    return faiss.read_index(str(INDEX_FILE))

# Load the DPR question tokenizer and encoder
def load_question_encoder():
    # The question encoder converts a user question into, the same vector space as the passage embeddings
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        QUESTION_ENCODER_MODEL
    )

    model = DPRQuestionEncoder.from_pretrained(
        QUESTION_ENCODER_MODEL
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device

# Convert the user question into a DPR embedding vector
def encode_question(question, tokenizer, model, device):
    encoded_input = tokenizer(
        question,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Move tokenized input to the same device as the model
    encoded_input = {
        key: value.to(device)
        for key, value in encoded_input.items()
    }

    # This is inference only, so gradients are not required
    with torch.no_grad():
        output = model(**encoded_input)
        question_embedding = output.pooler_output

    # FAISS expects float32 NumPy arrays
    question_embedding = (
        question_embedding
        .cpu()
        .numpy()
        .astype("float32")
    )

    return question_embedding


def retrieve_top_chunks(
    question_embedding,
    index,
    chunks,
    top_k=TOP_K,
):
    # Search FAISS for the passages with the highest dot-product similarity to the question embedding
    scores, indices = index.search(
        question_embedding,
        top_k,
    )

    results = []

    for rank, (score, index_position) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        # Skip invalid FAISS positions incase it will happend
        if index_position < 0:
            continue

        chunk = chunks[index_position].copy()

        chunk["rank"] = rank
        chunk["score"] = float(score)

        results.append(chunk)

    return results

# Print the retrieved chunks in a readable format
def display_results(question, results):
    print()
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)

    for result in results:
        print()
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Paper ID: {result['paper_id']}")
        print(f"Title: {result['title']}")
        print(f"Page: {result['page']}")
        print(f"Topic: {result['topic_tag']}")
        print(f"DOI: {result['doi']}")
        print()
        print("Text:")
        print(result["text"])
        print("-" * 80)


def main():
    # Expected command:
        # python src/retrieval/search.py "(user question)"
    if len(sys.argv) < 2:
        print(
            'Usage: python src/retrieval/search.py '
            '"<question>"'
        )
        return

    # Join all command-line words into one question
    question = " ".join(sys.argv[1:]).strip()

    print("Loading chunks...")
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    print("Loading FAISS index...")
    index = load_faiss_index()

    # The number of FAISS vectors should match the number of chunks
    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunks.jsonl do not match. "
            f"Index vectors: {index.ntotal}, "
            f"Chunks: {len(chunks)}"
        )

    print("Loading DPR question encoder...")
    tokenizer, model, device = load_question_encoder()

    print("Encoding question...")
    question_embedding = encode_question(
        question,
        tokenizer,
        model,
        device,
    )

    print(f"Searching Top-{TOP_K} chunks...")

    results = retrieve_top_chunks(
        question_embedding,
        index,
        chunks,
    )

    display_results(
        question,
        results,
    )


if __name__ == "__main__":
    main()