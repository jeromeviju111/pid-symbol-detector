import cv2

from src.inference.corrections_db import apply_corrections, apply_additions
from src.inference.detect import detect_symbols_local
from src.inference.ocr import detect_text_tiled
from src.inference.tag_matching import assign_tags
from src.inference.line_detection import (
    isolate_pipe_lines,
    detect_line_segments_from_mask,
    merge_duplicate_lines,
    assign_line_ids,
)
from src.inference.arrow_detection import get_flow_arrows_with_direction, detect_arrowheads_geometric, merge_arrow_sources
from src.inference.pixel_graph_connectivity import build_connectivity_map


def resize_if_too_large(image_path, max_dimension=5000):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    if max(h, w) <= max_dimension:
        return image_path

    scale = max_dimension / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    resized_path = image_path.rsplit(".", 1)[0] + "_resized.png"
    cv2.imwrite(resized_path, resized)
    return resized_path


def recompute_lines_and_connectivity(page_path, symbols, text_detections):
    """Shared by initial processing AND manual-edit recompute, so both
    paths use identical logic — no drift between the two."""
    line_mask, shape_mask = isolate_pipe_lines(page_path, detections=symbols, kernel_len=25)
    raw_lines = detect_line_segments_from_mask(line_mask, min_length=40)
    merged_lines = merge_duplicate_lines(raw_lines)
    labeled_lines = assign_line_ids(merged_lines, text_detections, max_distance=100)

    # model_arrows will be [] once the model is retrained without a
    # flow_arrow class (see get_flow_arrows_with_direction) - harmless,
    # merge_arrow_sources just falls back entirely to the geometric pass.
    model_arrows = get_flow_arrows_with_direction(page_path, symbols)
    geometric_arrows = detect_arrowheads_geometric(shape_mask, line_endpoints=merged_lines)
    flow_arrows = merge_arrow_sources(model_arrows, geometric_arrows)

    connectivity_map = build_connectivity_map(page_path, symbols, line_mask=line_mask)

    return labeled_lines, flow_arrows, connectivity_map


def process_full_page(page_path, document=None, page_num=None):
    page_path = str(page_path)
    page_path = resize_if_too_large(page_path, max_dimension=5000)

    symbol_detections = detect_symbols_local(page_path)
    text_detections = detect_text_tiled(page_path, confidence_threshold=0.25)
    combined_result = assign_tags(symbol_detections, text_detections, max_distance=80)

    if document:
        combined_result = apply_corrections(combined_result, document)
        if page_num is not None:
            combined_result = apply_additions(combined_result, document, page_num)

    labeled_lines, flow_arrows, connectivity_map = recompute_lines_and_connectivity(
        page_path, combined_result, text_detections
    )

    return {
        "symbols": combined_result,
        "text": text_detections,
        "lines": labeled_lines,
        "arrows": flow_arrows,
        "connectivity": connectivity_map,
    }