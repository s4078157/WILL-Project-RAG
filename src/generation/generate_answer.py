from pathlib import Path
import sys

# Third-party 
from dotenv import load_dotenv
from openai import OpenAI


# Add the project root so retrieval modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.search import (
    load_chunks,
    load_faiss_index,
    load_question_encoder,
    encode_question,
    retrieve_top_chunks,
)


# Load environment variables from the project .env file (API)
load_dotenv(PROJECT_ROOT / ".env")


# Generation configuration
MODEL = "gpt-5.6-terra" #fancyyyyyyyyyy
TOP_K = 5
MAX_OUTPUT_TOKENS = 700


def build_evidence_context(retrieved_chunks):
    # Convert retrieved chunks into a structured evidence block that the LLM can reference using source numbers
    evidence_sections = []

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        doi = chunk.get("doi") or "Not available"

        evidence = (
            f"[Source {source_number}]\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Paper ID: {chunk['paper_id']}\n"
            f"Title: {chunk['title']}\n"
            f"Page: {chunk['page']}\n"
            f"DOI: {doi}\n"
            f"Evidence:\n{chunk['text']}"
        )

        evidence_sections.append(evidence)

    return "\n\n".join(evidence_sections)


def generate_answer(question, retrieved_chunks):
    # Build the evidence that will be provided to the LLM.
    evidence_context = build_evidence_context(
        retrieved_chunks
    )

    client = OpenAI()

# This need to be modifed if during eval
    instructions = """
You are a research evidence assistant for non-expert users.

Your task is to explain scientific research clearly and accurately.

Follow these rules strictly:

1. Answer ONLY using the evidence provided to you.
2. Do not use outside knowledge to add unsupported claims.
3. Synthesize information across the provided sources when useful.
4. Explain technical findings in clear language that a general user can understand.
5. Preserve important limitations, conditions, and uncertainty from the research.
6. Do not invent statistics, participants, methods, conclusions, or causal claims.
7. Cite factual claims using [Source 1], [Source 2], etc.
8. If the available evidence is insufficient to answer the question, clearly say that the evidence is insufficient.
9. Do not claim that a source supports something unless that information appears in the provided evidence.
10. Give a concise but sufficiently detailed answer.
11. Structure every supported answer using exactly these sections:

Detailed Answer:
- Explain the evidence accurately and with enough detail.
- Include important limitations and conditions.
- Cite claims using [Source X].

Simple Summary:
- Summarize the explanation in 2–4 short sentences.
- Use simple everyday language suitable for a non-expert.
- Do not introduce any new information that was not already explained above.
- Preserve important limitations instead of oversimplifying them.
"""

    user_input = f"""
Question:
{question}

Retrieved evidence:
{evidence_context}

Using only the evidence above, answer the question for a non-expert reader.

Provide:
1. A detailed evidence-based explanation.
2. A short Simple Summary explaining the same conclusion in easier language.
"""

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=user_input,
        reasoning={
            "effort": "low"
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return response.output_text

# Display traceable sources separately from the generated answer
def display_sources(retrieved_chunks):
    print()
    print("=" * 80)
    print("SUPPORTING SOURCES")
    print("=" * 80)

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        doi = chunk.get("doi") or "Not available"

        print()
        print(f"[Source {source_number}]")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Paper: {chunk['title']}")
        print(f"Page: {chunk['page']}")
        print(f"DOI: {doi}")


def main():
    # Expected command:
        # python src/generation/generate_answer.py "(user question)"
    if len(sys.argv) < 2:
        print(
            'Usage: python src/generation/generate_answer.py '
            '"<question>"'
        )
        return

    question = " ".join(sys.argv[1:]).strip()

    print("Loading research chunks...")
    chunks = load_chunks()

    print("Loading FAISS index...")
    index = load_faiss_index()

    # Ensure the FAISS index matches the current chunk dataset
    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunks.jsonl do not match. "
            f"Index vectors: {index.ntotal}, "
            f"Chunks: {len(chunks)}"
        )

    print("Loading DPR question encoder...")
    tokenizer, model, device = load_question_encoder()

    print("Retrieving evidence...")

    question_embedding = encode_question(
        question,
        tokenizer,
        model,
        device,
    )

    retrieved_chunks = retrieve_top_chunks(
        question_embedding,
        index,
        chunks,
        top_k=TOP_K,
    )

    print("Generating grounded answer...")

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print()
    print(answer)

    display_sources(retrieved_chunks)


if __name__ == "__main__":
    main()