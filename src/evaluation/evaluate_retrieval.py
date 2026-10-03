from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.search import (
    load_chunks,
    load_faiss_index,
    load_question_encoder,
    encode_question,
    retrieve_top_chunks,
)

QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation" / "evaluation_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results.csv"

TOP_K = 5


def load_questions():
    questions = []

    with open(QUESTIONS_FILE, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            questions.append(row)

    return questions


def main():
    print("Loading evaluation questions...")
    questions = load_questions()

    print("Loading chunks...")
    chunks = load_chunks()

    print("Loading FAISS index...")
    index = load_faiss_index()

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index vectors ({index.ntotal}) do not match chunks ({len(chunks)})."
        )

    print("Loading DPR question encoder...")
    tokenizer, model, device = load_question_encoder()

    results = []

    for i, row in enumerate(questions, start=1):
        question_id = row["question_id"]
        question = row["question"]
        expected_paper_id = row["expected_paper_id"]

        print()
        print(f"Evaluating {question_id}: {question}")

        question_embedding = encode_question(
            question,
            tokenizer,
            model,
            device,
        )

        retrieved = retrieve_top_chunks(
            question_embedding,
            index,
            chunks,
            top_k=TOP_K,
        )

        retrieved_paper_ids = [
            chunk["paper_id"]
            for chunk in retrieved
        ]

        top1_paper_id = (
            retrieved_paper_ids[0]
            if retrieved_paper_ids
            else ""
        )

        top1_correct = (
            "Yes"
            if top1_paper_id == expected_paper_id
            else "No"
        )

        top5_hit = (
            "Yes"
            if expected_paper_id in retrieved_paper_ids
            else "No"
        )

        relevant_chunks_in_top5 = sum(
            1
            for paper_id in retrieved_paper_ids
            if paper_id == expected_paper_id
        )

        first_relevant_rank = ""

        for rank, paper_id in enumerate(
            retrieved_paper_ids,
            start=1,
        ):
            if paper_id == expected_paper_id:
                first_relevant_rank = rank
                break

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_paper_id": expected_paper_id,
                "top1_paper_id": top1_paper_id,
                "top1_correct": top1_correct,
                "top5_hit": top5_hit,
                "first_relevant_rank": first_relevant_rank,
                "relevant_chunks_in_top5": relevant_chunks_in_top5,
                "rank1_paper": retrieved_paper_ids[0] if len(retrieved_paper_ids) > 0 else "",
                "rank2_paper": retrieved_paper_ids[1] if len(retrieved_paper_ids) > 1 else "",
                "rank3_paper": retrieved_paper_ids[2] if len(retrieved_paper_ids) > 2 else "",
                "rank4_paper": retrieved_paper_ids[3] if len(retrieved_paper_ids) > 3 else "",
                "rank5_paper": retrieved_paper_ids[4] if len(retrieved_paper_ids) > 4 else "",
            }
        )

        print(f"  Expected paper: {expected_paper_id}")
        print(f"  Retrieved papers: {retrieved_paper_ids}")
        print(f"  Top-1 correct: {top1_correct}")
        print(f"  Top-5 hit: {top5_hit}")
        print(f"  Relevant chunks in Top-5: {relevant_chunks_in_top5}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "question_id",
        "question",
        "expected_paper_id",
        "top1_paper_id",
        "top1_correct",
        "top5_hit",
        "first_relevant_rank",
        "relevant_chunks_in_top5",
        "rank1_paper",
        "rank2_paper",
        "rank3_paper",
        "rank4_paper",
        "rank5_paper",
    ]

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
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()