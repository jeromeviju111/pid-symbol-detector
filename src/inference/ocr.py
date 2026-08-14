import cv2
import easyocr

reader = easyocr.Reader(["en"], gpu=True)


def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def tile_image_for_ocr(img, tile_size=1500, overlap=150):
    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] > 20 and tile.shape[1] > 20:
                tiles.append((tile, x, y))
    return tiles


def deduplicate_text(text_detections, iou_threshold=0.5):
    def iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    kept = []
    used = [False] * len(text_detections)
    for i, t1 in enumerate(text_detections):
        if used[i]:
            continue
        for j, t2 in enumerate(text_detections):
            if i == j or used[j]:
                continue
            if iou(t1["bbox"], t2["bbox"]) > iou_threshold and t1["text"] == t2["text"]:
                used[j] = True
        kept.append(t1)
        used[i] = True
    return kept


def detect_text_tiled(image_path, confidence_threshold=0.25):
    image_path = str(image_path)
    img = cv2.imread(image_path)
    img_enhanced = enhance_for_ocr(img)
    tiles = tile_image_for_ocr(img_enhanced, tile_size=1500, overlap=150)

    text_detections = []
    for tile_img, offset_x, offset_y in tiles:
        results = reader.readtext(
            tile_img,
            low_text=0.3,
            text_threshold=0.5,
            mag_ratio=1.5,
        )
        for bbox, text, confidence in results:
            if confidence < confidence_threshold:
                continue
            xs = [p[0] + offset_x for p in bbox]
            ys = [p[1] + offset_y for p in bbox]
            text_detections.append({
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "text": text.strip(),
                "confidence": confidence
            })

    return deduplicate_text(text_detections)


def detect_text_quick(image_path, confidence_threshold=0.25):
    """Faster, single-pass OCR fallback for CPU-only testing (no tiling)."""
    results = reader.readtext(str(image_path), low_text=0.3, text_threshold=0.5)
    text_detections = []
    for bbox, text, confidence in results:
        if confidence < confidence_threshold:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        text_detections.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text.strip()})
    return text_detections
