# Standard library
from pathlib import Path
import csv
import json
import re


# root Protject paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data DIR
CLEAN_DIR = PROJECT_ROOT / "data" / "processed" / "clean"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "papers.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "chunks.jsonl"


# Chunking configuration
# Keep chunks small enough for DPR while preserving enough context
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


# Paper loader function, Load paper-level metadata from papers.csv
def load_paper_metadata():
    # Returns:
    #     list[dict]: Metadata rows for all research papers
    papers = []

    with open(METADATA_FILE, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            papers.append(row)

    return papers

# Split cleaned paper text using page markers
def split_pages(text):
    # Expected markers:
    #     --- PAGE 1 ---
    #     --- PAGE 2 ---

    # Keeping each page separate ensures that every chunk can
    # be traced back to a specific page in the source paper

    # Args:
    #     text (str): Cleaned research-paper text

    # Returns:
    #     list[dict]: Each dictionary contains a page number and page text
    page_pattern = r"--- PAGE (\d+) ---"

    # re.split keeps the captured page number in the resulting list
    parts = re.split(page_pattern, text)

    pages = []

    # Example output from re.split:
    # [
    #   text_before_first_marker,
    #   "1",
    #   page_1_text,
    #   "2",
    #   page_2_text,
    # ]
    for index in range(1, len(parts), 2):
        page_number = int(parts[index])
        page_text = parts[index + 1].strip()

        if page_text:
            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                }
            )

    return pages

# Detect the start of the paper's bibliography
def remove_references(page_text):
    # The bibliography is excluded because reference-list text is not
    # primary evidence from the paper and may reduce retrieval quality

    # Args:
    #     page_text (str): Text from one PDF page

    # Returns:
    #     tuple[str, bool]:
    #         - Text before the REFERENCES heading
    #         - True if the REFERENCES section was found

    reference_match = re.search(
        r"(?m)^REFERENCES\s*$",
        page_text,
    )

    if reference_match:
        cleaned_text = page_text[:reference_match.start()].strip()
        return cleaned_text, True

    return page_text, False


def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    # Split page text into overlapping word-based chunks
    # Example:
    #     chunk_size = 220
    #     overlap = 40

    #     Chunk 1 -> words 1-220
    #     Chunk 2 -> words 181-400
    # Overlap helps preserve context when important information
    # falls near a chunk boundary

    # Args:
    #     text (str): Text from one page
    #     chunk_size (int): Maximum number of words per chunk
    #     overlap (int): Number of overlapping words between chunks

    # Returns:
    #     list[str]: List of chunk texts.

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    step_size = chunk_size - overlap
    start_index = 0

    while start_index < len(words):
        end_index = min(
            start_index + chunk_size,
            len(words),
        )

        chunk_words = words[start_index:end_index]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        # Stop when the final words have already been included.
        if end_index == len(words):
            break

        start_index += step_size

    return chunks


def process_paper(metadata):

    # Create chunks for one research paper

    # Each chunk receives paper-level metadata and the source page
    # so it can later be retrieved and cited by the RAG system

    # Args:
    #     metadata (dict): One row from papers.csv

    # Returns:
    #     list[dict]: Structured chunks for the paper

    paper_id = metadata["paper_id"]
    source_filename = metadata["source_filename"]

    input_path = CLEAN_DIR / source_filename

    if not input_path.exists():
        print(f"[WARNING] Cleaned file not found: {input_path}")
        return []

    paper_text = input_path.read_text(encoding="utf-8")
    pages = split_pages(paper_text)

    paper_chunks = []
    chunk_number = 1

    # Once REFERENCES is found, all following pages are excluded
    references_started = False

    for page in pages:
        if references_started:
            break

        page_number = page["page"]
        page_text = page["text"]

        # Remove bibliography text if REFERENCES begins on this page
        page_text, references_found = remove_references(page_text)

        if page_text:
            page_chunks = create_chunks(page_text)

            for chunk_text in page_chunks:
                chunk = {
                    "chunk_id": f"{paper_id}_C{chunk_number:03d}",
                    "paper_id": paper_id,
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "year": int(metadata["year"]),
                    "journal": metadata["journal"],
                    "page": page_number,
                    "doi": metadata["doi"],
                    "topic_tag": metadata["topic_tag"],
                    "text": chunk_text,
                }

                paper_chunks.append(chunk)
                chunk_number += 1

        if references_found:
            references_started = True

    return paper_chunks


def save_chunks(chunks):

    # Save all chunks as JSON Lines.

    # JSONL stores one JSON object per line, which is convenient for
    # later DPR embedding, indexing, debugging, and inspection
    # Args:
    #     chunks (list[dict]): All generated paper chunks

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    # Run the complete chunking pipeline
    # Pipeline:
    #     papers.csv
    #     + cleaned paper text
    #     -> page parsing
    #     -> chunking
    #     -> metadata attachment
    #     -> chunks.jsonl

    papers = load_paper_metadata()

    all_chunks = []

    for paper in papers:
        paper_id = paper["paper_id"]

        print(f"Processing {paper_id}...")

        paper_chunks = process_paper(paper)

        print(f"  Created {len(paper_chunks)} chunks")

        all_chunks.extend(paper_chunks)

    save_chunks(all_chunks)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()