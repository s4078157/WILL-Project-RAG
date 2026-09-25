from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DIR = PROCESSED_DIR / "clean"


def clean_text(text):
    # make the line endings sane
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove texrt download/access footer lines
    text = re.sub(
        r"^Downloaded from .*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove the repeated journal footer/header, to make it cleaner
    text = re.sub(
        r"^The Cumulative Cost of Additional Wakefulness—Van Dongen et alSLEEP.*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Join words broken across lines by PDF formatting, Example:
        # neurobe-
        # havioral
        # -> neurobehavioral
    text = re.sub(
        r"([A-Za-z])-\n([a-z])",
        r"\1\2",
        text
    )

    # Remove extra spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Remove trailing spaces on each line
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/ingestion/clean_text.py <txt_filename>")
        return

    txt_filename = sys.argv[1]
    input_path = PROCESSED_DIR / txt_filename

    if not input_path.exists():
        print(f"Text file not found: {input_path}")
        return

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    raw_text = input_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)

    output_path = CLEAN_DIR / txt_filename
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned: {txt_filename}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()