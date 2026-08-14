import numpy as np


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def assign_tags(detections, text_results, max_distance=80):
    used = set()
    for det in detections:
        center = bbox_center(det["bbox"])
        best_idx, best_dist = None, float("inf")
        for i, t in enumerate(text_results):
            if i in used:
                continue
            d = np.linalg.norm(np.array(center) - np.array(bbox_center(t["bbox"])))
            if d < best_dist and d <= max_distance:
                best_dist, best_idx = d, i
        det["tag"] = text_results[best_idx]["text"] if best_idx is not None else None
        if best_idx is not None:
            used.add(best_idx)
    return detections
