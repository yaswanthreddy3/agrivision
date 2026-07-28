import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config.settings import settings
from config.observability import setup_logfire
from retrieval.embeddings import get_embedding_dim

setup_logfire()

DOCS_COLLECTION = settings.QDRANT_COLLECTION  # "agrivision_docs"
OBSERVATIONS_COLLECTION = "agrivision_observations"

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        logfire.info("Qdrant client connected", url=settings.QDRANT_URL)
    return _client


def _ensure_collection(name: str, dim: int):
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        info = client.get_collection(name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != dim:
            raise ValueError(
                f"Qdrant collection '{name}' exists with dim={existing_dim}, but active model produces dim={dim}."
            )
        logfire.info("Collection already exists and matches dimension", collection=name, dim=dim)
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logfire.info("Collection created", collection=name, dim=dim)


def ensure_collection_exists():
    dim = get_embedding_dim()
    _ensure_collection(DOCS_COLLECTION, dim)


def ensure_observations_collection_exists():
    dim = get_embedding_dim()
    _ensure_collection(OBSERVATIONS_COLLECTION, dim)


COLLECTION_NAME = DOCS_COLLECTION  # backwards-compat alias


if __name__ == "__main__":
    ensure_collection_exists()
    ensure_observations_collection_exists()
    client = get_client()
    for name in [DOCS_COLLECTION, OBSERVATIONS_COLLECTION]:
        info = client.get_collection(name)
        print(f"Collection '{name}': vector_size={info.config.params.vectors.size}, points={info.points_count}")