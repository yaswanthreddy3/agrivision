# retrieval/observations.py
import uuid
import time
import logfire
from qdrant_client.models import PointStruct
from config.observability import setup_logfire
from retrieval.embeddings import embed_query, embed_texts
from retrieval.qdrant_client import get_client, ensure_observations_collection_exists, OBSERVATIONS_COLLECTION

setup_logfire()


def store_observation(caption_data: dict, image_path: str):
    """Stores a VLM analysis result into the observations collection for future retrieval."""
    ensure_observations_collection_exists()
    client = get_client()

    text_for_embedding = caption_data.get("raw_text", "")
    if not text_for_embedding:
        return

    vector = embed_texts([text_for_embedding])[0]

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "text": text_for_embedding,
            "crop": caption_data.get("crop"),
            "possible_disease": caption_data.get("possible_disease"),
            "severity": caption_data.get("severity"),
            "confidence": caption_data.get("confidence"),
            "symptoms": caption_data.get("symptoms", []),
            "image_path": image_path,
            "timestamp": time.time(),
        },
    )

    client.upsert(collection_name=OBSERVATIONS_COLLECTION, points=[point])
    logfire.info("observation_stored", crop=caption_data.get("crop"), disease=caption_data.get("possible_disease"))


def search_observations(query: str, top_k: int = 3) -> list[dict]:
    """Searches past observations relevant to a query. Returns empty list if none found."""
    client = get_client()
    query_vector = embed_query(query)

    try:
        results = client.query_points(
            collection_name=OBSERVATIONS_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    except Exception as e:
        logfire.warning(f"Observations search failed (collection may be empty): {e}")
        return []

    return [
        {
            "text": r.payload["text"],
            "source_file": f"past observation ({r.payload.get('crop', 'unknown crop')})",
            "page_number": None,
            "score": r.score,
            "confidence": r.payload.get("confidence"),
        }
        for r in results
    ]