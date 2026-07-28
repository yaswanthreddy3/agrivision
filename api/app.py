"""AgriVision Gradio + FastAPI app.

Run with: uvicorn api.app:app --reload
"""

import logging
import os
import time
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

from api.routes import router
from api.service import process_request

# --- Config (env-overridable, sane local defaults) --------------------------
STREAM_DELAY = float(os.getenv("STREAM_DELAY_SECONDS", "0.02"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agrivision")

# --- FastAPI ------------------------------------------------------------
app = FastAPI(title="AgriVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --- Gradio UI ------------------------------------------------------------
def _confidence_bar(conf: float) -> str:
    filled = round(conf * 10)
    return "🟩" * filled + "⬜" * (10 - filled)


def _extract_inputs(message) -> tuple[str | None, str | None]:
    """Pull (query, image_path) out of a Gradio multimodal message."""
    if isinstance(message, dict):
        query = message.get("text") or None
        files = message.get("files", [])
        image_path = None
        if files:
            first = files[0]
            image_path = first if isinstance(first, str) else first.get("path")
        return query, image_path
    if isinstance(message, str):
        return message, None
    return None, None


def predict(message, history, request: gr.Request):
    query, image_path = _extract_inputs(message)

    if not query and not image_path:
        yield "Please type a question or upload a leaf photo to get started."
        return

    if image_path and not Path(image_path).is_file():
        yield "⚠️ That image couldn't be read — please try uploading it again."
        return

    thread_id = request.session_hash if request else "default"

    try:
        result = process_request(query=query, image_path=image_path, thread_id=thread_id)
    except Exception:
        logger.exception("process_request failed | thread_id=%s query=%r", thread_id, query)
        yield "⚠️ Something went wrong on our end processing that request. Please try again."
        return

    final_answer = result.get("final_answer", "I couldn't generate a response for that.")

    header = ""
    captions = result.get("captions") or []
    if captions:
        c = captions[0]
        header = (
            f"#### 🔍 Detection: **{c['label']}**\n"
            f"{_confidence_bar(c['confidence'])} `{c['confidence']:.0%}` confidence\n\n---\n\n"
        )
        yield header

    streamed = header
    words = final_answer.split(" ")
    for i, word in enumerate(words):
        streamed += word + (" " if i < len(words) - 1 else "")
        yield streamed
        if STREAM_DELAY:
            time.sleep(STREAM_DELAY)


theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

demo = gr.ChatInterface(
    fn=predict,
    multimodal=True,
    title="🌱 AgriVision",
    description=(
        "**AI Crop Disease Assistant** — upload a leaf photo for instant diagnosis, "
        "or ask any crop health question. Answers are grounded in agricultural "
        "reference documents with citations."
    ),
    examples=[
        {"text": "What causes yellowing between leaf veins on tomato plants?"},
        {"text": "How do I treat powdery mildew organically?"},
        {"text": "What's the ideal soil pH for growing wheat?"},
    ],
    textbox=gr.MultimodalTextbox(
        placeholder="Ask a question or upload a leaf image...",
        file_types=["image"],
        sources=["upload"],
    ),
    chatbot=gr.Chatbot(height=520),
    submit_btn="Diagnose 🌿",
    cache_examples=False,
)

app = mount_gradio_app(app, demo, path="/app", theme=theme)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host=HOST, port=PORT, reload=os.getenv("RELOAD", "true") == "true")