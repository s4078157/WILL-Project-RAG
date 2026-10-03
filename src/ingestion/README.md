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

### Copy n Paste

Extract pdf `extract_pdfs.py`

```
python src\ingestion\extract_pdfs.py "[P1] Van Dongen et al., The Cumulative Cost of Additional Wakefulness.pdf"

python src\ingestion\extract_pdfs.py "[P2] Effects of a Single, Oral 60 mg Caffeine Dose on Attention in Healthy Adult Subjects.pdf"

python src\ingestion\extract_pdfs.py "[P3] Non-Visual Effects of Light on Melatonin, Alertness and Cognitive Performance.pdf"

python src\ingestion\extract_pdfs.py "[P4] Brain Drain The Mere Presence of One's Own Smartphone Reduces Available Cognitive Capacity.pdf"

python src\ingestion\extract_pdfs.py "[P5] Attention and short-term memory during occupational noise exposure considering task difficulty.pdf"

python src\ingestion\extract_pdfs.py "[P6] The Attention System of the Human Brain 20 Years After.pdf"

python src\ingestion\extract_pdfs.py "[P7] A comprehensive review of attention tests can we assess what we exactly do not understand.pdf"

python src\ingestion\extract_pdfs.py "[P8] The challenge of improving assessment and treatment for attention-deficit hyperactivity disorder in Australia.pdf"

python src\ingestion\extract_pdfs.py "[P9] Popular interventions to enhance sustained attention in children and adolescentsA critical systematic review☆.pdf"
```

clean text `clean_text.py`

```
python src\ingestion\clean_text.py "[P1] Van Dongen et al., The Cumulative Cost of Additional Wakefulness.txt"

python src\ingestion\clean_text.py "[P2] Effects of a Single, Oral 60 mg Caffeine Dose on Attention in Healthy Adult Subjects.txt"

python src\ingestion\clean_text.py "[P3] Non-Visual Effects of Light on Melatonin, Alertness and Cognitive Performance.txt"

python src\ingestion\clean_text.py "[P4] Brain Drain The Mere Presence of One's Own Smartphone Reduces Available Cognitive Capacity.txt"

python src\ingestion\clean_text.py "[P5] Attention and short-term memory during occupational noise exposure considering task difficulty.txt"

python src\ingestion\clean_text.py "[P6] The Attention System of the Human Brain 20 Years After.txt"

python src\ingestion\clean_text.py "[P7] A comprehensive review of attention tests can we assess what we exactly do not understand.txt"

python src\ingestion\clean_text.py "[P8] The challenge of improving assessment and treatment for attention-deficit hyperactivity disorder in Australia.txt"

python src\ingestion\clean_text.py "[P9] Popular interventions to enhance sustained attention in children and adolescentsA critical systematic review☆.txt"
```
