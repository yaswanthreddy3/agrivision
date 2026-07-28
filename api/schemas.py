from typing import Optional
from pydantic import BaseModel


class QueryResponse(BaseModel):
    input_type: Optional[str]
    answer: str
    detected_label: Optional[str] = None
    detected_confidence: Optional[float] = None
    session_id: str