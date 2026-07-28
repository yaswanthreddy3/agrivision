import logging
from typing import Any, Dict, List
from threading import Lock

import logfire
from flashrank import Ranker, RerankRequest
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class RerankSettings(BaseSettings):
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    default_top_k: int = 5

    class Config:
        env_prefix = "RERANK_"


settings = RerankSettings()

_ranker: Ranker | None = None
_ranker_lock = Lock()


def get_ranker() -> Ranker:
    """Lazily loads the FlashRank model once per process (double-checked locking)."""
    global _ranker
    if _ranker is None:
        with _ranker_lock:
            if _ranker is None:
                try:
                    with logfire.span("load_flashrank_model", model=settings.rerank_model):
                        _ranker = Ranker(model_name=settings.rerank_model)
                    logfire.info("FlashRank ranker loaded", model=settings.rerank_model)
                except Exception as e:
                    logger.error(f"Failed to load FlashRank model: {e}")
                    raise RuntimeError(f"Reranker model failed to load: {e}")
    return _ranker


def rerank(query: str, search_results: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
    """Reranks vector search results using a cross-encoder model."""
    if not search_results:
        return []

    k = top_k or settings.default_top_k
    ranker = get_ranker()

    passages = [
        {"id": f"idx_{i}", "text": result.get("text", "")}
        for i, result in enumerate(search_results)
    ]

    try:
        rerank_request = RerankRequest(query=query, passages=passages)
        with logfire.span("flashrank_rerank", query=query, num_candidates=len(search_results)):
            reranked = ranker.rerank(rerank_request)
    except Exception as e:
        # If the reranker fails, fall back to the original vector-search order
        # rather than failing the whole request.
        logfire.exception("FlashRank failed, falling back to vector-search order", error=str(e))
        logger.warning(f"Reranking error: {e}. Falling back.")
        return search_results[:k]

    id_to_original = {f"idx_{i}": result for i, result in enumerate(search_results)}

    final_results = []
    for item in reranked[:k]:
        passage_id = item.get("id")
        if passage_id in id_to_original:
            final_results.append({
                **id_to_original[passage_id],
                "rerank_score": float(item.get("score", 0.0)),
            })

    logfire.info(
        "rerank_complete",
        query=query,
        num_candidates=len(search_results),
        top_k_returned=len(final_results),
    )

    return final_results