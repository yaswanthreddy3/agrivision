from typing import TypedDict, Optional, List, Literal, Annotated
from langgraph.graph.message import add_messages


class PlannerState(TypedDict):
    image_path: Optional[str]
    query: Optional[str]
    input_type: Optional[Literal["image", "text", "unclear"]]
    detections: Optional[List[dict]]
    captions: Optional[List[dict]]
    fallback_caption: Optional[str]
    needs_clarification: bool
    clarification_question: Optional[str]
    retrieval_results: Optional[list]
    final_answer: Optional[str]
    guardrail_blocked: Optional[bool]
    messages: Annotated[List, add_messages]