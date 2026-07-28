# ARCHITECTURE

This document describes the system architecture and node responsibilities in detail.

## Node responsibilities

- Planner: classifies incoming requests (image/text/unclear), applies input guardrails, and routes execution to the detection or retrieval node. Keeps state minimal and explicit.

- Detection: runs a vision-capable LLM (VLM) to produce structured JSON describing crop, symptoms, candidate diagnoses, confidence scores, and image usability checks (single leaf vs grid). When confident, it may emit a retrieval query derived from the structured output.

- Retrieval: embeds queries, queries Qdrant (collections `agrivision_docs` and `agrivision_observations`), and returns candidate passages with provenance metadata.

- Reranker: FlashRank cross-encoder reranks the top candidate passages to improve precision prior to synthesis.

- Responder: decides final output path (clarify, synthesize, fallback), formats the response, and appends required source citations.

## Ingest and storage

- PDFs are parsed (pdfplumber, fallback pypdf), cleaned, chunked with overlap, and stored in Qdrant with metadata (filename, page).
- Observations coming from confident vision detections are stored in a separate collection with lower trust classification.

## Observability

- Node-level traces and timings are emitted via Logfire and optional LangSmith hooks for debugging and auditing.
