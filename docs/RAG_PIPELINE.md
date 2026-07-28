# RAG_PIPELINE

Detailed pipeline for retrieval, embeddings, chunking, and synthesis.

## Document ingestion

- PDF parsing: `pdfplumber` with `pypdf` fallback.
- Text cleaning and metadata preservation (filename, page).
- Chunking: recursive splitter with configurable chunk size and overlap.

## Embeddings

- Preferred: Gemini embeddings (3072‑d) via `langchain-google-genai.GoogleGenerativeAIEmbeddings`.
- Fallback: `sentence-transformers` (`all-mpnet-base-v2`) for local/offline use. The fallback can incur heavy init cost; codebase supports lazy init and a GEMINI_ONLY toggle.

## Vector search and reranking

- Qdrant stores chunk embeddings and metadata.
- Query embedding → nearest neighbors from Qdrant.
- FlashRank cross‑encoder reranker reorders top passages by relevance to the exact query.

## Synthesis and grounding

- The synthesizer (Groq) receives: system prompt, recent conversation history (short window), retrieved passages, and the user question — in that order.
- Synthesizer is constrained to produce answers only supported by retrieved passages and to list used sources at the end of responses.
- Observations from vision path are marked as lower trust and cannot be the sole evidence for treatment recommendations.
