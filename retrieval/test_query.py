from retrieval.embeddings import embed_query
from retrieval.qdrant_client import get_client, COLLECTION_NAME


def search(query: str, top_k: int = 5):
    client = get_client()
    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    print(f"\nQuery: {query!r}\n{'-'*60}")
    for i, r in enumerate(results, 1):
        text_preview = r.payload.get("text", "")[:200].replace("\n", " ")
        source = r.payload.get("source_file", "?")
        page = r.payload.get("page_number", "?")
        print(f"\n[{i}] score={r.score:.4f}  source={source}  page={page}")
        print(f"    {text_preview}...")


if __name__ == "__main__":
    # Pick a query relevant to your PDF content — adjust based on what's actually in it
    search("how to cure tamato leaf curl virus", top_k=10) 

