import cv2
import numpy as np
from src.inference.tag_matching import bbox_center


def find_symbol_at_click(detections, x, y, tolerance=20):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return d
    best, best_dist = None, float("inf")
    for d in detections:
        cx, cy = bbox_center(d["bbox"])
        dist = np.linalg.norm(np.array([x, y]) - np.array([cx, cy]))
        if dist < best_dist and dist <= tolerance + max(d["bbox"][2]-d["bbox"][0], d["bbox"][3]-d["bbox"][1]):
            best_dist, best = dist, d
    return best


def find_matches_across_pages(page_paths_list, all_page_data, target_class):
    results_per_page = {}
    for i, page_path in enumerate(page_paths_list, start=1):
        matches = [d for d in all_page_data[i]["symbols"] if d["class"] == target_class]
        if matches:
            results_per_page[i] = matches
    return results_per_page


def draw_page_matches(page_path, matches):
    img = cv2.imread(page_path)
    for d in matches:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d.get("tag") or "no ID"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
