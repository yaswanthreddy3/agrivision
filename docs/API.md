# API

FastAPI endpoints and payloads.

## Gradio demo

- The Gradio chat UI is mounted at `/app` in `api/app.py`.
- `predict()` in `api/app.py` adapts Gradio inputs (text or multimodal message) to the `process_request` service.

## REST endpoint

- `POST /query` (defined in `api/routes.py`) — multipart form submission supporting:
  - `query` (form field, optional)
  - `image` (file, optional)
  - `session_id` (form field, optional) — provide to persist session memory across requests

Response model: `api.schemas.QueryResponse` with fields: `input_type`, `answer`, `detected_label`, `detected_confidence`, `session_id`.

## Service

- `api.service.process_request()` delegates to `graph.build_graph().invoke(...)` and returns planner results.

## Notes

- If sending images programmatically, upload as multipart file with the `image` form field.
- To preserve conversational memory across requests, pass `session_id` from the previous response.
