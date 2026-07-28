import base64
import json
from typing import Any, Optional

import logfire
from langchain_groq import ChatGroq

from config.observability import setup_logfire
from config.settings import settings

setup_logfire()

VISION_MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You are an agricultural vision assistant analyzing a plant leaf image.

Your task is to validate whether the image is suitable for disease diagnosis before attempting any diagnosis.

Return ONLY valid JSON with this exact schema:
{
  "valid_image": true,
  "reason": "...",
  "crop": "...",
  "crop_confidence": 0.0,
  "symptoms": [],
  "diseased_organ": "...",
  "severity": "...",
  "possible_disease": "...",
  "confidence": 0.0,
  "visible_pests": [],
  "environmental_clues": [],
  "recommendation": null,
  "diagnosis_ready": true,
  "is_single_leaf": true
}

Rules:
1. First determine whether the image is suitable for diagnosis.
2. Reject ONLY if the image is clearly one of these — be conservative, when in doubt, ACCEPT the image:
   - a screenshot with visible UI elements (buttons, browser chrome, app interface)
   - an obvious multi-panel grid with hard borders separating distinct sub-images
   - a dataset visualization (labeled axes, charts, grids of thumbnails)
   - an annotated image with bounding boxes or overlaid text labels
3. A single natural photograph of a leaf is ALWAYS acceptable, even if it includes:
   - a hand or fingers holding the leaf
   - background clutter, other leaves, or parts of the plant
   - imperfect lighting, mild blur, or an off-center subject
   None of these are grounds for rejection. Only reject for the specific reasons in rule 2.
4. Never diagnose invalid images.
5. Never combine multiple unrelated leaves or plants into one diagnosis.
6. Never infer treatment.
7. If valid_image is false, set recommendation to exactly:
   "Please upload one clear photograph of a single affected plant leaf."
8. If valid_image is true:
   1. Identify the crop if possible.
   2. Describe ONLY visible symptoms.
   3. Estimate disease only if confidence >= 0.5.
   4. If confidence < 0.5, set possible_disease to null.
   5. Never guess.
   6. Never recommend treatment.
9. Use severity as one of: "mild", "moderate", or "severe".
10. Use crop_confidence as a float between 0.0 and 1.0 for how confident you are about the crop identification.
11. Set diagnosis_ready to true only when the image is clear enough to support a useful diagnosis.
12. Set is_single_leaf to true only when the image clearly focuses on a single leaf or a small single-plant region.
13. Return only JSON, no markdown, no commentary, no extra text.
"""

_llm: Optional[ChatGroq] = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=VISION_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=400,
            reasoning_effort="none",
        )
    return _llm


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.replace("```json", "").replace("```", "").strip()
    return text


def _default_result(*, parse_error: bool = False, reason: Optional[str] = None) -> dict[str, Any]:
    return {
        "valid_image": False,
        "reason": reason or "Image is not suitable for diagnosis.",
        "crop": None,
        "crop_confidence": 0.0,
        "symptoms": [],
        "diseased_organ": None,
        "severity": None,
        "possible_disease": None,
        "confidence": 0.0,
        "visible_pests": [],
        "environmental_clues": [],
        "recommendation": "Please upload one clear photograph of a single affected plant leaf.",
        "diagnosis_ready": False,
        "is_single_leaf": False,
        "parse_error": parse_error,
    }


def _normalize_result(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return _default_result(parse_error=True, reason="Model returned an unexpected response format.")

    valid_image = bool(parsed.get("valid_image", False))
    reason = parsed.get("reason") if isinstance(parsed.get("reason"), str) else None
    crop = parsed.get("crop") if isinstance(parsed.get("crop"), str) else None
    crop_confidence = parsed.get("crop_confidence")
    if not isinstance(crop_confidence, (int, float)):
        crop_confidence = 0.0
    crop_confidence = max(0.0, min(1.0, float(crop_confidence)))
    symptoms = parsed.get("symptoms") if isinstance(parsed.get("symptoms"), list) else []
    diseased_organ = parsed.get("diseased_organ") if isinstance(parsed.get("diseased_organ"), str) else None
    severity = parsed.get("severity")
    if severity not in {"mild", "moderate", "severe"}:
        severity = None
    possible_disease = parsed.get("possible_disease") if isinstance(parsed.get("possible_disease"), str) else None
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    visible_pests = parsed.get("visible_pests") if isinstance(parsed.get("visible_pests"), list) else []
    environmental_clues = parsed.get("environmental_clues") if isinstance(parsed.get("environmental_clues"), list) else []
    recommendation = parsed.get("recommendation") if isinstance(parsed.get("recommendation"), str) else None
    diagnosis_ready = bool(parsed.get("diagnosis_ready", False))
    is_single_leaf = bool(parsed.get("is_single_leaf", False))

    if not valid_image:
        return {
            "valid_image": False,
            "reason": reason or "Image is not suitable for diagnosis.",
            "crop": crop,
            "crop_confidence": crop_confidence,
            "symptoms": symptoms,
            "diseased_organ": diseased_organ,
            "severity": severity,
            "possible_disease": None,
            "confidence": float(confidence),
            "visible_pests": visible_pests,
            "environmental_clues": environmental_clues,
            "recommendation": recommendation or "Please upload one clear photograph of a single affected plant leaf.",
            "diagnosis_ready": False,
            "is_single_leaf": is_single_leaf,
        }

    if confidence < 0.5:
        possible_disease = None
        diagnosis_ready = False
        reason = "Disease cannot be identified confidently from the image."
        recommendation = "Please upload a closer, clearer image of the affected leaf."
    elif possible_disease:
        # Trust our own confidence threshold over the model's self-reported diagnosis_ready flag
        diagnosis_ready = True

    return {
        "valid_image": True,
        "reason": reason or "Image appears suitable for diagnosis.",
        "crop": crop,
        "crop_confidence": crop_confidence,
        "symptoms": symptoms,
        "diseased_organ": diseased_organ,
        "severity": severity,
        "possible_disease": possible_disease,
        "confidence": float(confidence),
        "visible_pests": visible_pests,
        "environmental_clues": environmental_clues,
        "recommendation": recommendation,
        "diagnosis_ready": diagnosis_ready,
        "is_single_leaf": is_single_leaf,
    }


def caption_leaf_image(image_path: str) -> dict[str, Any]:
    with logfire.span("vlm_captioning", crop_path=image_path):
        llm = get_llm()
        base64_image = encode_image(image_path)

        messages = [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                [
                    {"type": "text", "text": "Analyze this leaf image and return the JSON."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            ),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()
        cleaned = _strip_markdown_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logfire.error("caption_json_parse_failed", raw_response=raw, error=str(exc))
            parsed = _default_result(parse_error=True, reason="Unable to parse model response.")

        normalized = _normalize_result(parsed)
        if normalized.get("parse_error") is True:
            normalized.pop("parse_error", None)

        logfire.info(
            "caption_generated",
            valid_image=normalized.get("valid_image"),
            crop=normalized.get("crop"),
            possible_disease=normalized.get("possible_disease"),
            confidence=normalized.get("confidence"),
            severity=normalized.get("severity"),
        )

        return normalized


if __name__ == "__main__":
    import sys

    test_crop = sys.argv[1] if len(sys.argv) > 1 else "data/crops/sample.jpg"
    result = caption_leaf_image(test_crop)
    print(json.dumps(result, indent=2))
