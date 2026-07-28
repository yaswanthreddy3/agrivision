# EVALUATION

Notes on evaluation, testset generation, and metrics.

## Testset

- `eval/testset.json` is used by `eval/run_ragas.py` for RAG evaluation. Populate it with (query, expected_answer, expected_sources) pairs.
- `tests/Generatetestset.py` provides a helper for generating QA pairs from PDFs; review outputs before using them as ground truth.

## Metrics

- Precision@k and Recall@k for retrieval results.
- RAG faithfulness metrics (e.g., whether the synthesizer's claims are supported by returned passages) — evaluate via manual spot checks or automated assertion scripts against the sources.
- Reranker effectiveness measured via improvement in top‑k precision after reranking.

## Running evaluations

- Prepare `eval/testset.json`.
- Run `python eval/run_ragas.py` and review output metrics and logs.

## Recommendations

- Maintain a curated, small gold testset of real extension QA pairs for high‑quality evaluations.
- Use consistent preprocessing between ingest and evaluation to avoid tokenization mismatches.
