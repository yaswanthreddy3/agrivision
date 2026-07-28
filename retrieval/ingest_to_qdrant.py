import time
import uuid
import logfire
from qdrant_client.models import PointStruct
from config.observability import setup_logfire
from ingestion.parse_docs import process_all_pdfs, DocChunk
from retrieval.embeddings import embed_texts, BATCH_SIZE
from retrieval.qdrant_client import get_client, ensure_collection_exists, COLLECTION_NAME

setup_logfire()

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10


def make_point_id(chunk: DocChunk) -> str:
    """Deterministic ID from content — same chunk always maps to same ID,
    so re-running ingestion is safe (upsert overwrites instead of duplicating)."""
    key = f"{chunk.source_file}:{chunk.page_number}:{chunk.chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def chunk_to_point(chunk: DocChunk, vector: list[float]) -> PointStruct:
    return PointStruct(
        id=make_point_id(chunk),
        vector=vector,
        payload={
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        },
    )


def filter_already_ingested(chunks: list[DocChunk], client) -> list[DocChunk]:
    """Skip chunks already present in Qdrant, so resuming doesn't re-embed
    (and re-pay for) chunks that already succeeded."""
    all_ids = [make_point_id(c) for c in chunks]
    existing_ids = set()

    for i in range(0, len(all_ids), 500):
        batch_ids = all_ids[i : i + 500]
        existing = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids.update(p.id for p in existing)

    remaining = [c for c, pid in zip(chunks, all_ids) if pid not in existing_ids]
    skipped = len(chunks) - len(remaining)
    if skipped:
        print(f"  {skipped} chunks already ingested — skipping.")
    return remaining


def ingest_chunks(chunks: list[DocChunk]):
    if not chunks:
        print("No chunks to ingest.")
        return

    ensure_collection_exists()
    client = get_client()

    chunks = filter_already_ingested(chunks, client)
    if not chunks:
        print("Nothing to do — all chunks already ingested.")
        return

    total = len(chunks)
    print(f"Embedding + upserting {total} chunks in batches of {BATCH_SIZE}...")

    total_upserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch_chunks]

        with logfire.span("embed_and_upsert_batch", start=i, size=len(batch_chunks)):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    vectors = embed_texts(texts)
                    points = [chunk_to_point(c, v) for c, v in zip(batch_chunks, vectors)]
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    total_upserted += len(points)
                    print(f"  [{i + len(batch_chunks)}/{total}] embedded + upserted (running total: {total_upserted})")
                    break
                except Exception as e:
                    print(f"  ⚠️  Batch {i} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                    logfire.error("batch_failed", start=i, attempt=attempt, error=str(e))
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_WAIT_SECONDS)
                    else:
                        print(f"  ❌ Batch {i} permanently failed after {MAX_RETRIES} attempts — skipping. Re-run the script later to retry it.")

    logfire.info("ingestion_to_qdrant_complete", total_points=total_upserted, collection=COLLECTION_NAME)
    print(f"\n✅ Done. {total_upserted} chunks newly ingested into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    print("Step 1: Parsing PDFs into chunks...\n")
    chunks = process_all_pdfs()

    print(f"\nStep 2: Ingesting {len(chunks)} chunks into Qdrant...\n")
    ingest_chunks(chunks)

    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nFinal collection state — points count: {info.points_count}")