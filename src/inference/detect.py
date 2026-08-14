from ultralytics import YOLO
import torch
from torchvision.ops import nms

model = YOLO("models/best.pt")

CLASS_CONFIDENCE_OVERRIDES = {
    "flow_arrow": 0.03,
}
DEFAULT_CONFIDENCE = 0.05


def deduplicate_overlapping(detections, iou_threshold=0.4):
    if not detections:
        return []
    boxes = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["confidence"] for d in detections], dtype=torch.float32)
    keep_idx = nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_idx]


def detect_symbols_local(image_path, iou_threshold=0.4):
    # low model-level floor — lets everything through so our own
    # per-class filtering below can decide what actually survives
    results = model.predict(str(image_path), conf=0.02, imgsz=1024, agnostic_nms=True)

    detections = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)

        threshold = CLASS_CONFIDENCE_OVERRIDES.get(cls_name, DEFAULT_CONFIDENCE)
        if conf < threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

    return deduplicate_overlapping(detections, iou_threshold)