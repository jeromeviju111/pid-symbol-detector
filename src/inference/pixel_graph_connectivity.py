import cv2
import numpy as np
from collections import deque


def build_binary_mask(image_path, close_kernel=7):
    """Fallback only — used if no pre-masked line_mask is provided."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def bbox_edge_points(bbox, step=5):
    x1, y1, x2, y2 = map(int, bbox)
    points = []
    for x in range(x1, x2, step):
        points.append((x, y1))
        points.append((x, y2))
    for y in range(y1, y2, step):
        points.append((x1, y))
        points.append((x2, y))
    return points


def find_connected_symbols(binary_mask, source_index, detections, max_steps=100000):
    if source_index < 0 or source_index >= len(detections):
        return {}
    h, w = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    parent = {}
    queue = deque()

    neighbors8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def sample_line_start_points(bbox, margin=20):
        x1, y1, x2, y2 = map(int, bbox)
        points = []
        for y in range(y1 - margin, y2 + margin + 1):
            for x in range(x1 - margin, x2 + margin + 1):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    continue
                points.append((x, y))
        return points

    def point_is_adjacent_to_bbox(x, y, bbox, margin=16):
        bx1, by1, bx2, by2 = map(int, bbox)
        return (
            bx1 - margin <= x <= bx2 + margin and
            by1 - margin <= y <= by2 + margin and
            not (bx1 <= x <= bx2 and by1 <= y <= by2)
        )

    def enqueue_nearby_line_pixels(bbox, margin=20):
        bx1, by1, bx2, by2 = map(int, bbox)
        for y in range(max(0, by1 - margin), min(h, by2 + margin + 1)):
            for x in range(max(0, bx1 - margin), min(w, bx2 + margin + 1)):
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    continue
                if not visited[y, x] and binary_mask[y, x] > 0:
                    queue.append((x, y))
                    visited[y, x] = True
                    parent[(x, y)] = None
                    return True
        return False

    start_points = sample_line_start_points(detections[source_index]["bbox"])
    for pt in start_points:
        x, y = pt
        if 0 <= x < w and 0 <= y < h and not visited[y, x]:
            if binary_mask[y, x] > 0:
                queue.append((x, y))
                visited[y, x] = True
                parent[(x, y)] = None
            else:
                for dx, dy in neighbors8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and binary_mask[ny, nx] > 0:
                        queue.append((nx, ny))
                        visited[ny, nx] = True
                        parent[(nx, ny)] = None
                        break

    if not queue:
        enqueue_nearby_line_pixels(detections[source_index]["bbox"], margin=20)

    if not queue:
        bx1, by1, bx2, by2 = map(int, detections[source_index]["bbox"])
        for y in range(max(0, by1 - 32), min(h, by2 + 32) + 1):
            for x in range(max(0, bx1 - 32), min(w, bx2 + 32) + 1):
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    continue
                if not visited[y, x] and binary_mask[y, x] > 0:
                    queue.append((x, y))
                    visited[y, x] = True
                    parent[(x, y)] = None
                    break
            if queue:
                break

    if not queue:
        bx1, by1, bx2, by2 = map(int, detections[source_index]["bbox"])
        for y in range(max(0, by1 - 12), min(h, by2 + 12) + 1):
            for x in range(max(0, bx1 - 12), min(w, bx2 + 12) + 1):
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    continue
                if not visited[y, x] and binary_mask[y, x] > 0:
                    queue.append((x, y))
                    visited[y, x] = True
                    parent[(x, y)] = None
                    break
            if queue:
                break

    # Last-resort: if still no seeds, allow starting from pixels inside
    # the symbol bbox itself (some masking strategies erase the border
    # pixels and leave traces only inside). This is conservative: only
    # used when every other seeding method failed.
    if not queue:
        bx1, by1, bx2, by2 = map(int, detections[source_index]["bbox"])
        for y in range(max(0, by1 - 2), min(h, by2 + 2) + 1):
            for x in range(max(0, bx1 - 2), min(w, bx2 + 2) + 1):
                if not visited[y, x] and binary_mask[y, x] > 0:
                    queue.append((x, y))
                    visited[y, x] = True
                    parent[(x, y)] = None
                    break
            if queue:
                break

    connections = {}
    steps = 0

    while queue and steps < max_steps:
        x, y = queue.popleft()
        steps += 1

        hit_symbol = None
        for i, det in enumerate(detections):
            if i == source_index or i in connections:
                continue
            bx1, by1, bx2, by2 = det["bbox"]
            if (bx1 <= x <= bx2 and by1 <= y <= by2) or point_is_adjacent_to_bbox(x, y, det["bbox"]):
                hit_symbol = i
                break

        if hit_symbol is not None:
            path = [(x, y)]
            cur = (x, y)
            while parent.get(cur) is not None:
                cur = parent[cur]
                path.append(cur)
            connections[hit_symbol] = list(reversed(path))
            continue

        for dx, dy in neighbors8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and binary_mask[ny, nx] > 0:
                visited[ny, nx] = True
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))

    return connections


def build_connectivity_map(image_path, detections, line_mask=None):
    """If line_mask is provided (the symbol-masked mask from isolate_pipe_lines),
    reuse it, but bridge small pixel gaps first — junctions, corners, and
    anti-aliasing artifacts can leave single-pixel breaks that stop the
    pixel-by-pixel BFS even on otherwise-solid lines."""
    if line_mask is not None:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, close_kernel)
    else:
        binary_mask = build_binary_mask(image_path)

    connectivity = {}
    for i in range(len(detections)):
        connectivity[i] = find_connected_symbols(binary_mask, i, detections)
    return connectivity