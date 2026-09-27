from pathlib import Path
import json

# Third-party libraries, our choice based on the WALERT 
import faiss
import numpy as np
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


# Project root paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chunking, Embedding, and Indexes Dir path
CHUNKS_FILE = PROJECT_ROOT / "data" / "metadata" / "chunks.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "target" / "embeddings"
INDEX_DIR = PROJECT_ROOT / "target" / "indexes"

# Target file outcome
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "dpr_passage_embeddings.npy"
INDEX_FILE = INDEX_DIR / "dpr_faiss.index"


# DPR model configuration
CONTEXT_ENCODER_MODEL = "facebook/dpr-ctx_encoder-multiset-base"

# Small batch size, because we run it with the CPU (based on Alex laptop capability)
BATCH_SIZE = 4

# DPR is based on BERT and supports up to 512 tokens
MAX_LENGTH = 512


# Load structured research chunks from chunks.jsonl
def load_chunks():
    # Returns:
    #     list[dict]: All chunks in the same order that will be
    #     used for embedding and FAISS indexing.

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks

# Load the DPR context tokenizer and context encoder
def load_encoder():
    # The context encoder converts research-paper chunks into, dense numerical vectors
    # Returns:
    #     tuple:
    #         tokenizer
    #         model
    #         device

    # this will check if the device use GPU, if no use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    #tokenizer
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        CONTEXT_ENCODER_MODEL
    )

    model = DPRContextEncoder.from_pretrained(
        CONTEXT_ENCODER_MODEL
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device

# Encode all research chunks using the DPR context encoder
def encode_chunks(chunks, tokenizer, model, device):
    # DPR uses a paper title together with the chunk text as context

    # Args:
    #     chunks (list[dict]): Research chunks.
    #     tokenizer: DPR context tokenizer.
    #     model: DPR context encoder.
    #     device: CPU or CUDA device.

    # Returns:
    #     np.ndarray: Passage embeddings with shape:
    #                 (number_of_chunks, embedding_dimension)
    embeddings = []

    total_chunks = len(chunks)

    for start_index in range(0, total_chunks, BATCH_SIZE):
        end_index = min(
            start_index + BATCH_SIZE,
            total_chunks,
        )

        batch = chunks[start_index:end_index]

        # DPR context encoding benefits from using both
        # the paper title and the passage text
        titles = [
            chunk["title"]
            for chunk in batch
        ]
        texts = [
            chunk["text"]
            for chunk in batch
        ]

        encoded_input = tokenizer(
            titles,
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Move tokenized input to the same device as the model.
        encoded_input = {
            key: value.to(device)
            for key, value in encoded_input.items()
        }

        # Disable gradient calculation because this is inference, not model training
        with torch.no_grad():
            output = model(**encoded_input)

            batch_embeddings = output.pooler_output

        # FAISS expects float32 NumPy arrays
        batch_embeddings = (
            batch_embeddings
            .cpu()
            .numpy()
            .astype("float32")
        )

        embeddings.append(batch_embeddings)

        print(
            f"Encoded {end_index}/{total_chunks} chunks"
        )

    return np.vstack(embeddings)

# Build a FAISS inner-product index
def build_faiss_index(embeddings):
    # DPR retrieval uses dot-product similarity between question embeddings and passage embeddings

    # Args:
    #     embeddings (np.ndarray): Passage vectors

    # Returns:
    #     faiss.IndexFlatIP: Searchable FAISS index

    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(embeddings)

    return index

# Saving the passage embeddings and FAISS index locally
def save_outputs(embeddings, index):
    # Generated embeddings and indexes are not committed to GitHub, because they can be reproduced from the pipeline (also because some copyright issue)
    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )


# Run the DPR passage-indexing pipeline
def main():
    # Pipeline:
    #     chunks.jsonl
    #     -> DPR Context Encoder
    #     -> passage embeddings
    #     -> FAISS index
    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    if not chunks:
        raise ValueError(
            "No chunks found in chunks.jsonl."
        )

    print()
    print("Loading DPR context encoder...")

    tokenizer, model, device = load_encoder()

    print()
    print("Encoding passages...")

    embeddings = encode_chunks(
        chunks,
        tokenizer,
        model,
        device,
    )

    print()
    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("Building FAISS index...")

    index = build_faiss_index(
        embeddings
    )

    save_outputs(
        embeddings,
        index,
    )

    print()
    print("Indexing complete.")
    print(
        f"FAISS vectors: {index.ntotal}"
    )
    print(
        f"Embeddings saved to: {EMBEDDINGS_FILE}"
    )
    print(
        f"FAISS index saved to: {INDEX_FILE}"
    )


if __name__ == "__main__":
    main()