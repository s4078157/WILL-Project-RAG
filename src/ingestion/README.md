## Ingestion

The folder has scripts for preparing research papers before they are retrieved

### Files

#### `extract_pdfs.py`

Extracts text from a PDF and keeps the page numbers.

Input:

```text
data/raw/
````

Output:

```text
data/processed/
```

Example:

```bash
python src/ingestion/extract_pdfs.py "[P1] Van Dongen et al., The Cumulative Cost of Additional Wakefulness.pdf"
```

---

#### `clean_text.py`

Cleans the extracted text.

It removes some PDF formatting problems such as:

* broken words
* unnecessary spaces
* repeated download information
* repeated journal footer text

Input:

```text
data/processed/
```

Output:

```text
data/processed/clean/
```

Example:

```bash
python src/ingestion/clean_text.py "[P1] Van Dongen et al., The Cumulative Cost of Additional Wakefulness.txt"
```

---

#### `chunk_text.py`

The cleaned papers are split up into smaller chunks for the retrieval system

Current settings:

```text
Chunk size: 220 words
Overlap: 40 words
```

It also adds metadata such as:

```text
chunk_id
paper_id
title
authors
year
journal
page
doi
topic_tag
text
```

Input:

```text
data/processed/clean/
data/metadata/papers.csv
```

Output:

```text
data/metadata/chunks.jsonl
```

Run:

```bash
python src/ingestion/chunk_text.py
```

---

### Ingestion Flow

```text
PDF
↓
extract_pdfs.py
↓
Extracted Text
↓
clean_text.py
↓
Cleaned Text
↓
chunk_text.py
↓
chunks.jsonl
```

### Important

The original raw PDFs and the text extracted from them are kept locally and should not be uploaded to the public GitHub repository. However, this situation may be altered in the future since the decision currently remains under the authority of RAG

