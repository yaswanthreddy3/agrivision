from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import PlannerState
from graph.planner_node import classify_input_node, route_input
from graph.detection_node import caption_node
from graph.retrieval_node import retrieve_node
from graph.responder_node import responder_node
from config.observability import setup_logfire
import logfire

setup_logfire()

_checkpointer = MemorySaver()


def build_graph():
    graph = StateGraph(PlannerState)

    graph.add_node("classify", classify_input_node)
    graph.add_node("caption", caption_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("respond", responder_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify", route_input,
        {"respond": "respond", "detect": "caption", "retrieve": "retrieve"}
    )
    graph.add_conditional_edges(
        "caption",
        lambda state: "retrieve" if not state.get("needs_clarification") else "respond",
        {"retrieve": "retrieve", "respond": "respond"}
    )

    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=_checkpointer)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    image_path = None
    query = None

    if len(args) == 1:
        query = args[0]
    elif len(args) == 2:
        image_path = args[0] if args[0] else None
        query = args[1] if args[1] else None

    app = build_graph()
    config = {"configurable": {"thread_id": "cli-test-session"}}

    with logfire.span("planner_run", image_path=image_path, query=query):
        result = app.invoke({
            "image_path": image_path,
            "query": query,
            "needs_clarification": False
        }, config=config)

    print("\n--- Result ---")
    print(f"Input type: {result.get('input_type')}")
    print(f"\nFinal answer:\n{result['final_answer']}")