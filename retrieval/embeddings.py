import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings
from config.observability import setup_logfire

setup_logfire()

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2
_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────
def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return
    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────
def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────
def _extract_retry_delay(err_str: str) -> int | None:
    """Pull the server-suggested retryDelay (seconds) out of a Gemini error string."""
    import re
    match = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
    return int(match.group(1)) + 2 if match else None  # +2s buffer


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Up to 6 attempts. Honors Google's own retryDelay when given,
        # otherwise falls back to exponential backoff capped at 60s.
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e)
                err_lower = err.lower()
                is_rate_limit = any(x in err_lower for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < max_attempts - 1:
                    wait = _extract_retry_delay(err) or min(2 ** attempt, 60)
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/{max_attempts})."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API ──────────────────────────────────────────────────────────────
def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
        if _model_type == "gemini" and i + BATCH_SIZE < len(texts):
            time.sleep(5)  # breathing room between requests, avoid RPM burst
    return all_embeddings 