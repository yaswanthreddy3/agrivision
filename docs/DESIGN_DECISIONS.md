# DESIGN_DECISIONS

This document consolidates engineering decisions, highlights, and key learnings.

## Major decisions

- LangGraph node split: keeps concerns separate (image diagnosis, retrieval, reranking, synthesis).
- Gemini embeddings preferred for dimensionality and managed service; local `sentence-transformers` fallback for offline work.
- Qdrant chosen for vector storage and metadata provenance.
- FlashRank cross‑encoder for reranking to improve precision.
- Groq used for vision-capable synthesis where deterministic outputs and controlled prompts are required.

## Engineering highlights & learnings

- Typed `PlannerState`, small node interfaces, and clear side effects make unit testing straightforward.
- Reranking materially improves synthesizer input quality; prioritize integrating cross‑encoder rerankers in evaluation.
- Strict system prompts and explicit grounding reduce hallucinations.
- Session memory using `MessagesState` provides pragmatic context without replacing document evidence.
- Observability via Logfire and LangSmith is essential for post‑hoc audits and troubleshooting.
