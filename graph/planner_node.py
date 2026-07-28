from langchain_core.messages import HumanMessage
from graph.state import PlannerState
from guardrails.rails import guard
from graph.query_contextualizer import contextualize_query


def classify_input_node(state: PlannerState) -> PlannerState:
    text_to_check = state.get("query")

    if text_to_check:
        # Resolve follow-up questions using conversation history BEFORE guardrails/retrieval
        history = state.get("messages", [])
        resolved_query = contextualize_query(text_to_check, history)
        state["query"] = resolved_query

        blocked, refusal_message = guard(resolved_query)
        if blocked:
            state["input_type"] = "unclear"
            state["needs_clarification"] = True
            state["clarification_question"] = refusal_message
            state["guardrail_blocked"] = True
            state["messages"] = [HumanMessage(content=text_to_check)]
            return state

        state["messages"] = [HumanMessage(content=text_to_check)]

    has_image = bool(state.get("image_path"))
    has_query = bool(state.get("query"))

    if has_image:
        state["input_type"] = "image"
    elif has_query:
        state["input_type"] = "text"
    else:
        state["input_type"] = "unclear"
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "I didn't receive an image or a question — could you upload a leaf photo, "
            "or tell me what you'd like to know?"
        )
    return state


def route_input(state: PlannerState) -> str:
    if state["input_type"] == "unclear":
        return "respond"
    elif state["input_type"] == "image":
        return "detect"
    else:
        return "retrieve"