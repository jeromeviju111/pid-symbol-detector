import cv2
import numpy as np


def find_tip_from_triangle_vertices(contour):
    """For small/simple triangle shapes, find the tip directly from the
    3 approximated vertices — more reliable than PCA on tiny contours,
    where spread-based statistics can be unstable."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

    if len(approx) != 3:
        return None, None

    pts = approx.reshape(-1, 2).astype(np.float64)

    best_tip = None
    best_dist = -1
    for i in range(3):
        tip_candidate = pts[i]
        other_two = np.delete(pts, i, axis=0)
        base_mid = other_two.mean(axis=0)
        dist = np.linalg.norm(tip_candidate - base_mid)
        if dist > best_dist:
            best_dist = dist
            best_tip = tip_candidate

    center = pts.mean(axis=0)
    tip_vector = best_tip - center
    return center, tip_vector


def find_tip_via_pca(contour):
    """Fallback for shapes that aren't clean triangles — finds the tip
    using the shape's main axis and spread at each end, instead of
    trusting a single farthest pixel."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    side_axis = eigvecs[:, np.argmin(eigvals)]

    proj_main = centered @ main_axis
    proj_side = centered @ side_axis

    lo, hi = proj_main.min(), proj_main.max()
    span = hi - lo
    if span == 0:
        return None, None

    low_mask = proj_main <= lo + 0.2 * span
    high_mask = proj_main >= hi - 0.2 * span

    low_spread = np.std(proj_side[low_mask]) if low_mask.any() else float("inf")
    high_spread = np.std(proj_side[high_mask]) if high_mask.any() else float("inf")

    tip_direction = main_axis if high_spread < low_spread else -main_axis
    return mean, tip_direction


def vector_to_compass(dx, dy):
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"


def detect_arrow_direction_from_shape(image_path, bbox):
    """For arrows the model already detected (class == flow_arrow)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # strip thin connecting stems (a few px wide) so only the solid
    # triangle body remains — prevents a stem sliver from skewing
    # which point gets identified as the "tip"
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5:
        return None

    center, tip_vector = find_tip_from_triangle_vertices(largest)

    if tip_vector is None:
        center, tip_vector = find_tip_via_pca(largest)

    if tip_vector is None:
        return None

    return vector_to_compass(tip_vector[0], tip_vector[1])

def get_flow_arrows_with_direction(image_path, detections):
    arrows = []
    for d in detections:
        if d["class"] != "flow_arrow":
            continue
        direction = detect_arrow_direction_from_shape(image_path, d["bbox"])
        if direction:
            arrows.append({"bbox": d["bbox"], "direction": direction})
    return arrows


def _triangle_elongation_and_tip(pts):
    """Rotation-invariant elongation (long-axis / short-axis via PCA) plus
    the tip point, for a 3-vertex approx polygon. Using PCA instead of a
    plain bbox aspect ratio matters here because bbox aspect ratio depends
    on how the triangle happens to sit relative to the image axes — a
    genuine arrowhead pointing at 45 degrees has a near-square bbox and
    would otherwise get wrongly discarded as "not elongated enough"."""
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    elongation = np.sqrt(eigvals[-1] / eigvals[0])

    best_tip, best_dist = None, -1
    for i in range(3):
        tip_candidate = pts[i]
        other_two = np.delete(pts, i, axis=0)
        base_mid = other_two.mean(axis=0)
        dist = np.linalg.norm(tip_candidate - base_mid)
        if dist > best_dist:
            best_dist = dist
            best_tip = tip_candidate

    return elongation, best_tip


def detect_arrowheads_geometric(shape_mask, line_endpoints=None, min_area=12,
                                 max_area=500, search_radius=25, min_elongation=1.15):
    """Finds arrowhead shapes near the ends of detected pipe lines.

    IMPORTANT: pass the `shape_mask` returned by isolate_pipe_lines, NOT the
    line-only mask. The line mask is built with long directional (25px)
    morphological opening that keeps straight runs and erases everything
    else — including a filled arrowhead triangle a few pixels wide, which
    is never a 25px-long straight run in any single orientation. shape_mask
    only has a light 3x3 close applied, so compact blobs like arrowheads
    survive.

    If line_endpoints is given (list of (x1,y1,x2,y2) lines), candidates are
    only searched for within `search_radius` px of a line's endpoints. This
    cuts false positives dramatically (valve bowties, junction dots, text
    fragments elsewhere on the page) and lets each arrow be tied to the
    specific line it terminates. If line_endpoints is None, falls back to a
    full-mask scan.
    """
    contours, _ = cv2.findContours(shape_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) != 3:
            continue

        pts = approx.reshape(-1, 2).astype(np.float64)
        elongation, tip = _triangle_elongation_and_tip(pts)
        if elongation < min_elongation:
            continue

        x, y, w, h = cv2.boundingRect(c)
        center = pts.mean(axis=0)
        tip_vector = tip - center
        direction = vector_to_compass(tip_vector[0], tip_vector[1])
        candidates.append({"bbox": [x, y, x + w, y + h], "direction": direction, "center": center})

    if not line_endpoints:
        return [{"bbox": a["bbox"], "direction": a["direction"]} for a in candidates]

    endpoints = [pt for line in line_endpoints for pt in ((line[0], line[1]), (line[2], line[3]))]
    arrows = []
    for a in candidates:
        close_to_a_line = any(
            np.linalg.norm(a["center"] - np.array(ep)) <= search_radius for ep in endpoints
        )
        if close_to_a_line:
            arrows.append({"bbox": a["bbox"], "direction": a["direction"]})
    return arrows


def merge_arrow_sources(model_arrows, geometric_arrows, iou_threshold=0.3):
    """Combines both detection methods, removing duplicates where
    both found the same arrow."""
    def iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    combined = list(model_arrows)
    for g in geometric_arrows:
        if not any(iou(g["bbox"], m["bbox"]) > iou_threshold for m in model_arrows):
            combined.append(g)
    return combined
