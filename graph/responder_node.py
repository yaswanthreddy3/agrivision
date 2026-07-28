import logfire
from langchain_core.messages import AIMessage
from graph.state import PlannerState
from retrieval.synthesize import synthesize_answer


def responder_node(state: PlannerState) -> PlannerState:
    with logfire.span("responder_node"):
        if state.get("needs_clarification"):
            state["final_answer"] = state["clarification_question"]
        elif state.get("retrieval_results"):
            state["final_answer"] = synthesize_answer(state["query"], state["retrieval_results"])
        else:
            state["final_answer"] = "I wasn't able to process this request. Please try again."

        state["messages"] = [AIMessage(content=state["final_answer"])]
        logfire.info("response_finalized", answer_preview=state["final_answer"][:100])

    return state