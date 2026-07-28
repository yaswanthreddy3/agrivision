from graph.build_graph import build_graph

graph = build_graph()


def process_request(query=None, image_path=None, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {
            "query": query,
            "image_path": image_path,
            "needs_clarification": False,
        },
        config=config,
    )