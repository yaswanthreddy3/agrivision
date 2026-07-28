import logfire
from graph.state import PlannerState
from detection.captioning import caption_leaf_image
from retrieval.observations import store_observation

LOW_CONFIDENCE_THRESHOLD = 0.45


def _build_summary_text(caption_data: dict) -> str:
    crop = caption_data.get("crop") or "unknown crop"
    symptoms = caption_data.get("symptoms") or []
    symptom_str = ", ".join(symptoms) if symptoms else "no specific symptoms listed"
    disease = caption_data.get("possible_disease")
    if disease:
        return f"Observed on {crop}: symptoms including {symptom_str}, consistent with possible {disease}."
    return f"Observed on {crop}: symptoms including {symptom_str}."


def caption_node(state: PlannerState) -> PlannerState:
    with logfire.span("vlm_analysis", image_path=state["image_path"]):
        caption_data = caption_leaf_image(state["image_path"])

        # Reject invalid images (collages, screenshots, multi-plant, etc.) immediately
        if not caption_data.get("valid_image", False):
            state["captions"] = []
            state["needs_clarification"] = True
            state["clarification_question"] = (
                caption_data.get("recommendation")
                or "Please upload one clear photograph of a single affected plant leaf."
            )
            return state

        summary_text = _build_summary_text(caption_data)
        caption_data["raw_text"] = summary_text  # keep for backwards-compat with observations.py

        store_observation(caption_data, state["image_path"])

        possible_disease = caption_data.get("possible_disease")
        confidence = caption_data.get("confidence", 0.0)
        diagnosis_ready = caption_data.get("diagnosis_ready", False)
        is_uncertain = not diagnosis_ready or possible_disease is None or confidence < LOW_CONFIDENCE_THRESHOLD

        state["captions"] = [{
            "crop_path": state["image_path"],
            "label": possible_disease or "unidentified",
            "confidence": confidence,
            "caption": summary_text,
            "caption_structured": caption_data,
            "label_uncertain": is_uncertain,
        }]

        if is_uncertain:
            state["needs_clarification"] = True
            reason = caption_data.get("reason", "")
            state["clarification_question"] = (
                f"I'm not fully confident, but here's what I observe: {summary_text} "
                f"{reason} Could you confirm the crop or provide a clearer image?"
            )
        else:
            state["needs_clarification"] = False
            state["query"] = f"how to treat {possible_disease}"

    return state