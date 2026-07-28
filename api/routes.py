import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import logfire
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.schemas import QueryResponse
from api.service import process_request

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def run_query(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
):
    """
    Supports:
    1. Text only
    2. Image only
    3. Image + Text

    session_id (optional): pass the same value across requests to maintain
    conversation memory via LangGraph's MemorySaver checkpointer. If omitted,
    a fresh UUID is generated (no memory across separate calls).
    """

    if not query and image is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a query, an image, or both.",
        )

    thread_id = session_id or str(uuid.uuid4())
    image_path = None

    try:
        if image is not None:
            suffix = Path(image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(image.file, tmp)
                image_path = tmp.name

        with logfire.span("planner_run", image_path=image_path, query=query, thread_id=thread_id):
            result = process_request(query=query, image_path=image_path, thread_id=thread_id)

        captions = result.get("captions") or []
        detected = (
            captions[0]
            if captions and not result.get("needs_clarification")
            else None
        )

        return QueryResponse(
            input_type=result.get("input_type"),
            answer=result.get("final_answer"),
            detected_label=detected["label"] if detected else None,
            detected_confidence=detected["confidence"] if detected else None,
            session_id=thread_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)