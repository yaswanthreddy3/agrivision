# DEPLOYMENT

Guidance for deploying AgriVision in a production‑inspired environment.

## Requirements

- Python 3.11+ virtual environment
- Qdrant instance (cloud or self‑hosted) with network access
- API keys for cloud services (Gemini, Groq) stored in environment variables

## Recommended setup

1. Containerize the app with Docker; expose port `8000` for the FastAPI app.
2. Use a process manager (systemd / Kubernetes) and run multiple worker replicas for horizontal scale.
3. Use an external Redis or Postgres for persistent session memory (MemorySaver) if cross‑instance sessions are required.
4. Secure credentials via environment variables or a secrets manager; never commit keys to source.
5. Monitor node latency and model API errors via Logfire/LangSmith traces.

## Environment variables (examples)

- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `QDRANT_CLUSTER_ENDPOINT`
- `QDRANT_API_KEY`
- `LOGFIRE_TOKEN` (optional)
- `LANGSMITH_API_KEY` (optional)

## Notes


- Reranker services (FlashRank) may be served separately as a microservice for efficiency.
