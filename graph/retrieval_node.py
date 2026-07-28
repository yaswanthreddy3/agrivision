import logfire
from graph.state import PlannerState
from retrieval.embeddings import embed_query
from retrieval.qdrant_client import get_client, COLLECTION_NAME
from retrieval.reranker import rerank
from retrieval.observations import search_observations


def retrieve_node(state: PlannerState) -> PlannerState:
    query = state["query"]

    with logfire.span("retrieve_node", query=query):
        query_vector = embed_query(query)
        client = get_client()

        doc_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=10,
            with_payload=True,
        ).points

        search_results = [
            {
                "text": r.payload["text"],
                "source_file": r.payload["source_file"],
                "page_number": r.payload["page_number"],
                "score": r.score,
            }
            for r in doc_results
        ]

        obs_results = search_observations(query, top_k=3)
        combined = search_results + obs_results

        reranked = rerank(query, combined, top_k=5)
        state["retrieval_results"] = reranked

    return state
