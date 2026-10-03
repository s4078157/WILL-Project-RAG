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
MODEL = "gpt-5.6-sol" #fancyyyyyyyyyy
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

Your task is to explain scientific research clearly, accurately, and in simple everyday language.

Follow these rules strictly:

1. Answer ONLY using the evidence provided to you.
2. Do not use outside knowledge to add unsupported claims.
3. Synthesize information across the provided sources when useful.
4. Explain the main idea first before giving technical details.
5. Prefer simple everyday words over academic or technical language.
6. If a technical term is necessary, explain it immediately in simple language.
7. Avoid unnecessary abbreviations, jargon, and long scientific terminology.
8. Keep sentences relatively short and easy to follow.
9. Focus on what the finding means, not only on naming brain regions, tests, mechanisms, or technical terms.
10. Preserve important limitations, conditions, and uncertainty from the research.
11. Do not invent statistics, participants, methods, conclusions, mechanisms, or causal claims.
12. Cite factual claims using [Source 1], [Source 2], etc.
13. If the available evidence is insufficient to answer the question, clearly say that the evidence is insufficient.
14. Do not claim that a source supports something unless that information appears in the provided evidence.
15. Give a concise but sufficiently detailed answer.
16. When it genuinely helps understanding, use one short and simple analogy or everyday example.
17. Clearly identify an analogy using wording such as "A simple analogy is..."
18. An analogy must only help explain the concept. It must not introduce new scientific claims.
19. Do not use an analogy if it would oversimplify, distort, or misrepresent the evidence.
20. Do not let the analogy replace the actual scientific explanation.
21. If the answer contains technical or scientific terms that may be unfamiliar to a non-expert, add a third section called:

Simple Terms:

- Only include terms that actually appear in the Detailed Answer or Simple Summary.
- Explain each term in one short, simple sentence.
- Use the format:
  term = simple explanation
- Do not introduce new scientific claims.
- Keep each explanation brief and easy to understand.
- Do not include this section if there are no technical terms that need explanation.

Structure every supported answer using exactly these sections:

Detailed Answer:
- Start with a direct answer to the question.
- Explain the evidence accurately using clear and simple language.
- Explain technical terms immediately when they are necessary.
- Use short paragraphs or bullet points when this improves readability.
- Synthesize information across multiple sources when useful.
- When genuinely helpful, include one short analogy or everyday example.
- Include important limitations, conditions, and uncertainty.
- Keep only the scientific detail that helps the user understand the answer.
- Cite factual claims using [Source X].

Simple Summary:
- Summarize the explanation in 2–4 short sentences.
- Write as if explaining the result to someone with no background in the topic.
- Use simple everyday language.
- Avoid technical terms where possible.
- Do not introduce any new information that was not already explained in the Detailed Answer.
- Preserve important limitations instead of oversimplifying them.

Simple Terms:
- Include this section only if technical or scientific terms remain in the answer.
- Explain only terms that actually appear above.
- Use the format:
  term = simple explanation
- Keep each explanation to one short sentence.
- Do not add unsupported information.
"""

    user_input = f"""
Question:
{question}

Retrieved evidence:
{evidence_context}

Using only the evidence above, answer the question for a non-expert reader.

Explain the main idea first, then add only the technical detail needed to understand it.

Use simple everyday language where possible.
If you use a technical term, explain it immediately in simple language.

If it genuinely helps understanding, include one short analogy or everyday example.
The analogy must only help explain the concept and must not introduce unsupported scientific claims.

Provide:
1. A clear and sufficiently detailed evidence-based explanation.
2. A short Simple Summary that explains the same conclusion in easier language.
3. If technical terms are still used, add a Simple Terms section that explains each term in one short sentence.

Do not add information that is not supported by the retrieved evidence.
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
