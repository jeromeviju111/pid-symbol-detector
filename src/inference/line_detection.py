import cv2
import numpy as np
from src.inference.tag_matching import bbox_center


def isolate_pipe_lines(image_path, detections=None, kernel_len=25, symbol_margin=3, dash_kernel_len=6):
    """Detects line-like structures, masking out known symbol regions first
    so lines aren't drawn through valve/pump/box interiors.

    Runs two passes: a long-kernel pass for solid lines, and a short-kernel
    pass for dashed/signal lines (which get erased by the long-kernel opening
    since individual dashes are shorter than kernel_len), then bridges the
    gaps between dashes so they read as one continuous connected line.

    The dash-bridging step can dilate real line pixels back into the
    symbol regions we just blanked, so every blanked region is re-applied
    at the end to guarantee no bleed-through survives.

    Returns (line_mask, shape_mask). line_mask keeps only long straight
    runs (what you want for LSD/connectivity). shape_mask is the
    symbol-blanked binary BEFORE that directional opening is applied, with
    only a small isotropic close to remove speckle noise — small compact
    blobs like arrowheads survive here, whereas they get erased by the
    line_mask's 25px directional kernels (a filled triangle a few pixels
    across is not a 25px-long straight run in any single direction).
    """
    image_path = str(image_path)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

    blank_regions = []

    if detections:
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            x1m = max(0, x1 - symbol_margin)
            y1m = max(0, y1 - symbol_margin)
            x2m = min(binary.shape[1], x2 + symbol_margin)
            y2m = min(binary.shape[0], y2 + symbol_margin)
            binary[y1m:y2m, x1m:x2m] = 0
            blank_regions.append((y1m, y2m, x1m, x2m))

    shape_speckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    shape_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, shape_speckle_kernel)
    for y1, y2, x1, x2 in blank_regions:
        shape_mask[y1:y2, x1:x2] = 0

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    solid_lines = cv2.bitwise_or(horiz, vert)

    dash_horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dash_kernel_len, 1))
    dash_vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dash_kernel_len))
    dash_horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, dash_horiz_kernel)
    dash_vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, dash_vert_kernel)
    dash_lines = cv2.bitwise_or(dash_horiz, dash_vert)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dash_lines_bridged = cv2.morphologyEx(dash_lines, cv2.MORPH_CLOSE, close_kernel)

    combined = cv2.bitwise_or(solid_lines, dash_lines_bridged)

    for y1, y2, x1, x2 in blank_regions:
        combined[y1:y2, x1:x2] = 0

    return combined, shape_mask


def detect_line_segments_from_mask(line_mask, min_length=40):
    lsd = cv2.createLineSegmentDetector(0)
    lines_raw, _, _, _ = lsd.detect(line_mask)
    lines = []
    if lines_raw is not None:
        for l in lines_raw:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= min_length:
                lines.append((int(x1), int(y1), int(x2), int(y2)))
    return lines


def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def line_midpoint(line):
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def merge_duplicate_lines(lines, angle_tol=5, dist_tol=10):
    merged = []
    used = [False] * len(lines)
    for i, l1 in enumerate(lines):
        if used[i]:
            continue
        angle1 = line_angle(*l1)
        mid1 = line_midpoint(l1)
        group = [l1]
        used[i] = True
        for j, l2 in enumerate(lines):
            if used[j] or i == j:
                continue
            angle2 = line_angle(*l2)
            mid2 = line_midpoint(l2)
            if abs(angle1 - angle2) < angle_tol and np.linalg.norm(np.array(mid1) - np.array(mid2)) < dist_tol:
                group.append(l2)
                used[j] = True
        xs = [p for l in group for p in (l[0], l[2])]
        ys = [p for l in group for p in (l[1], l[3])]
        merged.append((min(xs), min(ys), max(xs), max(ys)))
    return merged


def line_center(line):
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def assign_line_ids(lines, text_results, max_distance=100):
    labeled_lines = []
    for line in lines:
        center = line_center(line)
        best_text, best_dist = None, float("inf")
        for t in text_results:
            d = np.linalg.norm(np.array(center) - np.array(bbox_center(t["bbox"])))
            if d < best_dist and d <= max_distance:
                best_dist, best_text = d, t["text"]
        labeled_lines.append({"line": line, "line_id": best_text})
    return labeled_lines