from pathlib import Path
from pypdf import PdfReader
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    output_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        output_parts.append(
            f"\n--- PAGE {page_number} ---\n\n{text.strip()}\n"
        )

    return "\n".join(output_parts)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/ingestion/extract_pdfs.py <pdf_filename>")
        return

    pdf_filename = sys.argv[1]
    pdf_path = RAW_DIR / pdf_filename

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    text = extract_pdf(pdf_path)

    output_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
    output_path.write_text(text, encoding="utf-8")

    print(f"Extracted: {pdf_path.name}")
    print(f"Pages: {len(PdfReader(pdf_path).pages)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()