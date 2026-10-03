from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (
    load_rag_components,
    run_rag,
)

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_questions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "answer_results.csv"
)


def load_questions():
    questions = []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            questions.append(row)

    return questions


def main():
    print("Loading evaluation questions...")
    questions = load_questions()

    print("Loading RAG components...")
    components = load_rag_components()

    results = []

    for i, row in enumerate(
        questions,
        start=1,
    ):
        question_id = row["question_id"]
        question = row["question"]
        expected_paper_id = row["expected_paper_id"]

        print()
        print("=" * 80)
        print(
            f"Running {question_id} "
            f"({i}/{len(questions)})"
        )
        print(question)
        print("=" * 80)

        result = run_rag(
            question,
            components=components,
        )

        retrieved_chunks = result["retrieved_chunks"]

        retrieved_paper_ids = [
            chunk["paper_id"]
            for chunk in retrieved_chunks
        ]

        retrieved_chunk_ids = [
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        ]

        retrieved_pages = [
            str(chunk["page"])
            for chunk in retrieved_chunks
        ]

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_paper_id": expected_paper_id,
                "generated_answer": result["answer"],
                "retrieved_paper_ids": " | ".join(
                    retrieved_paper_ids
                ),
                "retrieved_chunk_ids": " | ".join(
                    retrieved_chunk_ids
                ),
                "retrieved_pages": " | ".join(
                    retrieved_pages
                ),
                "answer_correctness": "",
                "faithful_to_evidence": "",
                "source_correctness": "",
                "limitations_preserved": "",
                "failure_category": "",
                "review_notes": "",
            }
        )

        print("Completed.")

    fieldnames = [
        "question_id",
        "question",
        "expected_paper_id",
        "generated_answer",
        "retrieved_paper_ids",
        "retrieved_chunk_ids",
        "retrieved_pages",
        "answer_correctness",
        "faithful_to_evidence",
        "source_correctness",
        "limitations_preserved",
        "failure_category",
        "review_notes",
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"Saved answer evaluation data to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()