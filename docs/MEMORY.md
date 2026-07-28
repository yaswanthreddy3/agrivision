# MEMORY

Notes on session memory and message handling.

- Implementation: LangGraph `MessagesState` (in‑process MemorySaver pattern).
- Scope: last 4–6 turns per session to keep prompts compact and control cost.
- Purpose: pragmatic context (pronoun resolution, follow‑ups, preserving prior diagnosis) — not a factual authority.
- Storage: in‑process by default; production deployment should use a persistent saver (Redis/Postgres) to survive restarts and multi‑instance setups.
- Privacy: memory contains only message role and content; do not store sensitive environment variables or tokens.
