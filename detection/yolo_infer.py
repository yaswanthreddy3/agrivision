import onnxruntime as ort
import numpy as np
import cv2
import os
import uuid
import logfire

ONNX_MODEL_PATH = "models/best.onnx"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
IMG_SIZE = 640
CROP_OUTPUT_DIR = "data/crops"

CLASS_NAMES = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf',
    'Bell_pepper leaf', 'Bell_pepper leaf spot',
    'Corn - Healthy', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf',
    'Potato - Healthy', 'Potato leaf early blight', 'Potato leaf late blight',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
    'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus',
    'Tomato leaf yellow virus', 'Tomato mold leaf',
    'grape leaf', 'grape leaf black rot'
]

_session = None

def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session

def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img, img_batch

def postprocess(output, orig_shape, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores > conf_threshold
    predictions = predictions[mask]
    scores = scores[mask]

    if len(predictions) == 0:
        return []

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4]

    orig_h, orig_w = orig_shape[:2]
    scale_x, scale_y = orig_w / IMG_SIZE, orig_h / IMG_SIZE

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )

    results = []
    for i in np.array(indices).flatten():
        results.append({
            "bbox": boxes_xyxy[i].astype(int).tolist(),
            "confidence": float(scores[i]),
            "class_id": int(class_ids[i])
        })
    return results

def run_inference(image_path: str, conf_threshold: float = CONF_THRESHOLD) -> list[dict]:
    with logfire.span("yolo_inference", image_path=image_path):
        session = get_session()
        os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)

        orig_img, input_tensor = preprocess(image_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})

        detections_raw = postprocess(output, orig_img.shape, conf_threshold)

        detections = []
        for det in detections_raw:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)

            label = CLASS_NAMES[det["class_id"]]
            crop = orig_img[y1:y2, x1:x2]
            crop_filename = f"{uuid.uuid4().hex[:8]}_{det['class_id']}.jpg"
            crop_path = os.path.join(CROP_OUTPUT_DIR, crop_filename)
            cv2.imwrite(crop_path, crop)

            detections.append({
                "label": label,
                "confidence": det["confidence"],
                "bbox": [x1, y1, x2, y2],
                "crop_path": crop_path,
                "flagged": "healthy" not in label.lower()
            })

        logfire.info("detection_complete", num_detections=len(detections), detections=detections)
        return detections


if __name__ == "__main__":
    import sys
    test_image = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/sample.jpg"
    results = run_inference(test_image)
    for d in results:
        print(d)