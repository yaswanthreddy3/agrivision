# DATASET

Notes on document ingestion and dataset expectations.

## Source documents

- Place PDFs under `data/docs/` or `data/data/raw_pdfs/` before running ingestion.
- Documents should be authoritative extension materials (government reports, extension bulletins) to maximize trustworthiness.

## Ingestion

- Run `python -m retrieval.ingest_to_qdrant` to parse PDFs, chunk text, embed chunks, and upload to Qdrant.
- Chunks preserve metadata (`filename`, `page`) for provenance in responses.

## Dataset quality

- Prefer curated, high‑quality PDFs with clear pagination and minimal scanned/noisy text.
- For scanned PDFs, perform OCR preprocessing before ingestion.

## Storage

- Qdrant collections used:
  - `agrivision_docs` — primary document corpus
  - `agrivision_observations` — lower‑trust observations from confident vision detections
