import traceback
from pathlib import Path
import cv2
import gradio as gr
import pandas as pd
import numpy as np

from src.utils.pdf_convert import pdf_to_images
from src.inference.pipeline import process_full_page, recompute_lines_and_connectivity
from src.inference.visualize import VIEW_FUNCTIONS, draw_symbols_only, draw_connectivity_click_view
from src.inference.pixel_graph_connectivity import build_connectivity_map
from src.inference.matching import find_symbol_at_click, find_matches_across_pages, draw_page_matches
from src.inference.summary import build_symbol_summary_table
from src.inference.asset_hierarchy import build_asset_hierarchy_table
from src.inference.diagram_graph import build_diagram_graph, graph_to_text, try_answer_count_question
from src.inference.llm_query import ask_diagram_question, ask_diagram_question_openrouter
from src.inference.tag_matching import bbox_center
from src.inference.corrections_db import (
    remove_correction, save_correction, save_addition, remove_addition,
    list_corrections, get_additions_for_document,
)

app_state = {"page_paths": [], "page_data": {}}


def handle_upload(file_path):
    try:
        print(f"DEBUG: upload received, file_path = {file_path}")
        if file_path is None:
            return gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), "Waiting for file..."

        file_path = str(file_path)

        if file_path.lower().endswith(".pdf"):
            page_paths = pdf_to_images(file_path)
        else:
            page_paths = [file_path]

        page_data = {}
        # Stable key for correction/addition lookups (see corrections_db.py)
        # — the uploaded file's own name, not the full path Gradio hands
        # back, since that path lives in a fresh temp location on every
        # upload and would never match itself again on re-upload.
        document_id = Path(file_path).stem
        app_state["document_id"] = document_id

        for i, path in enumerate(page_paths, start=1):
            print(f"DEBUG: processing page {i}/{len(page_paths)}")
            page_data[i] = process_full_page(path, document=document_id, page_num=i)
            print(f"DEBUG: page {i} -> {len(page_data[i]['symbols'])} symbols, {len(page_data[i]['text'])} text")

        app_state["page_paths"] = page_paths
        app_state["page_data"] = page_data

        page_choices = [f"Page {i}" for i in range(1, len(page_paths) + 1)]
        print(f"DEBUG: done, {len(page_choices)} page(s) ready")

        return (
            gr.Dropdown(choices=page_choices, value=page_choices[0]),
            gr.Dropdown(choices=page_choices, value=page_choices[0]),
            gr.Dropdown(choices=page_choices, value=page_choices[0]),
            gr.Dropdown(choices=page_choices, value=page_choices[0]),
            gr.Dropdown(choices=page_choices, value=page_choices[0]),
            f"Loaded {len(page_paths)} page(s).",
        )
    except Exception as e:
        print(traceback.format_exc())
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), f"ERROR: {str(e)}"


def update_detection_view(page_choice, view_type):
    if not page_choice or not app_state["page_paths"]:
        return None, pd.DataFrame(columns=["Symbol Class", "Count"])
    page_num = int(page_choice.split(" ")[1])
    page_path = app_state["page_paths"][page_num - 1]
    page_data = app_state["page_data"][page_num]
    img = VIEW_FUNCTIONS[view_type](page_path, page_data)
    summary_table = build_symbol_summary_table(page_data["symbols"])
    return img, summary_table


def update_matcher_view(page_choice):
    if not page_choice or not app_state["page_paths"]:
        return None, [], "", pd.DataFrame(columns=["Symbol Class", "Count"])
    page_num = int(page_choice.split(" ")[1])
    page_path = app_state["page_paths"][page_num - 1]
    page_data = app_state["page_data"][page_num]
    summary_table = build_symbol_summary_table(page_data["symbols"])
    return draw_symbols_only(page_path, page_data), [], "", summary_table


def handle_symbol_click(page_choice, evt: gr.SelectData):
    try:
        x, y = evt.index
        page_num = int(page_choice.split(" ")[1])
        detections = app_state["page_data"][page_num]["symbols"]

        clicked = find_symbol_at_click(detections, x, y)
        if clicked is None:
            return [], "No symbol found near that click."

        target_class = clicked["class"]
        results_per_page = find_matches_across_pages(
            app_state["page_paths"], app_state["page_data"], target_class
        )

        gallery_images = []
        summary_lines = [f"Symbol selected: {target_class}\n"]
        for pnum, matches in results_per_page.items():
            page_path = app_state["page_paths"][pnum - 1]
            img = draw_page_matches(page_path, matches)
            gallery_images.append((img, f"Page {pnum} ({len(matches)} found)"))
            ids = [m.get("tag") or "no ID" for m in matches]
            summary_lines.append(f"Page {pnum}: {len(matches)} match(es) — IDs: {', '.join(ids)}")

        return gallery_images, "\n".join(summary_lines)
    except Exception as e:
        print(traceback.format_exc())
        return [], f"ERROR: {str(e)}"


def handle_reset(page_choice):
    return update_matcher_view(page_choice)


def _extract_role_content(message):
    """chat_history entries may come back as plain dicts OR as ChatMessage-
    like objects depending on Gradio version - handle both so history from
    a prior turn doesn't crash the next question. Content is coerced to a
    plain string: some Gradio versions attach extra structure (content
    blocks, metadata) to assistant messages rather than a raw string, and
    Ollama's /api/chat rejects non-string content with a 400."""
    if isinstance(message, dict):
        role, content = message.get("role"), message.get("content")
    else:
        role, content = getattr(message, "role", None), getattr(message, "content", None)

    if content is not None and not isinstance(content, str):
        if isinstance(content, list):
            # Content-block style: [{"type": "text", "text": "..."}]
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                else:
                    parts.append(str(block))
            content = " ".join(p for p in parts if p)
        else:
            content = str(content)

    return role, content


# ============================================================
# ASK THE DIAGRAM (local LLM chat over the extracted graph)
# ============================================================

def handle_diagram_chat(page_choice, question, chat_history, backend_choice):
    # Normalize immediately - whatever shape Gradio handed back (plain
    # dicts on turn 1, possibly ChatMessage-like objects on later turns),
    # work with a consistent list of dicts from here on.
    normalized_history = []
    for m in (chat_history or []):
        role, content = _extract_role_content(m)
        if role and content is not None:
            normalized_history.append({"role": role, "content": content})
    chat_history = normalized_history

    if not page_choice or not app_state.get("page_paths"):
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "Please upload and select a page first."})
        return chat_history, ""

    if not question or not question.strip():
        return chat_history, ""

    page_num = int(page_choice.split(" ")[1])
    page_data = app_state["page_data"][page_num]

    try:
        graph = build_diagram_graph(page_data, page_label=page_choice)
        graph_text = graph_to_text(graph)

        # Count questions are answered directly from the exact counts -
        # guaranteed correct regardless of model size, no LLM call needed.
        direct_answer = try_answer_count_question(question, graph.get("counts", {}))
        if direct_answer:
            answer = direct_answer
        elif backend_choice == "OpenRouter (hosted, paid)":
            answer = ask_diagram_question_openrouter(question, graph_text, history=chat_history)
        else:
            answer = ask_diagram_question(question, graph_text, history=chat_history)
    except Exception as e:
        print(traceback.format_exc())
        answer = f"Something went wrong building the answer: {e}"

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, ""


def handle_diagram_chat_page_change(page_choice):
    # Switching pages starts a fresh conversation - the graph fed to the LLM
    # is page-specific, so carrying old history across would confuse it.
    return []


# ============================================================
# MANUAL VERIFICATION — core drawing function
# ============================================================

def draw_manual_view(page_path, symbols, selected_index=None, highlight_bbox=None):
    img = cv2.imread(page_path)

    for i, d in enumerate(symbols):
        x1, y1, x2, y2 = map(int, d["bbox"])

        if i == selected_index:
            color = (0, 0, 255)
        elif d.get("manually_added"):
            color = (0, 140, 255)
        elif d.get("manually_verified"):
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        label = d["class"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if highlight_bbox:
        x1, y1, x2, y2 = map(int, highlight_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def update_manual_verification_view(page_choice):
    if not page_choice or not app_state.get("page_paths"):
        return None

    page_num = int(page_choice.split(" ")[1])
    if page_num > len(app_state["page_paths"]):
        return None

    page_path = app_state["page_paths"][page_num - 1]
    symbols = app_state["page_data"][page_num]["symbols"]
    return draw_manual_view(page_path, symbols)


def manual_verification_click(page_choice, verification_mode, evt: gr.SelectData):
    try:
        if not page_choice:
            return "Please select a page first.", None

        x, y = evt.index
        page_num = int(page_choice.split(" ")[1])
        symbols = app_state["page_data"][page_num]["symbols"]
        page_path = app_state["page_paths"][page_num - 1]

        if verification_mode == "Correct / Delete Symbol":
            clicked = find_symbol_at_click(symbols, x, y)
            if clicked is None:
                return "No detected symbol found at this click position.", draw_manual_view(page_path, symbols)

            selected_index = symbols.index(clicked)
            app_state["manual_selected_index"] = selected_index
            app_state["manual_active_page_num"] = page_num

            preview = draw_manual_view(page_path, symbols, selected_index=selected_index)
            return (
                f"Selected symbol: {clicked['class']}\n"
                f"• Enter a new class name and click 'Apply Class Correction', OR\n"
                f"• Click 'Delete Selected Symbol' to remove it.",
                preview
            )

        elif verification_mode == "Add Missing Symbol":
            if "manual_first_point" not in app_state:
                app_state["manual_first_point"] = (x, y)
                preview = draw_manual_view(page_path, symbols)
                return f"First corner set at ({x}, {y}). Now click the opposite corner.", preview

            x1, y1 = app_state.pop("manual_first_point")
            x2, y2 = x, y
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

            app_state["manual_new_bbox"] = bbox
            app_state["manual_active_page_num"] = page_num

            preview = draw_manual_view(page_path, symbols, highlight_bbox=bbox)
            return (
                f"Bounding box created: {bbox}\n"
                f"Enter a class name and click 'Add Missing Symbol'.",
                preview
            )

    except Exception as e:
        print(traceback.format_exc())
        return f"ERROR: {str(e)}", None


def apply_manual_class_correction(page_choice, corrected_class):
    try:
        if not corrected_class or not corrected_class.strip():
            return "Please enter a verified symbol class name.", update_manual_verification_view(page_choice)

        page_num = app_state.get("manual_active_page_num")
        selected_index = app_state.get("manual_selected_index")

        if page_num is None or selected_index is None:
            return "No existing symbol selected. Click a detected symbol first.", update_manual_verification_view(page_choice)

        symbols = app_state["page_data"][page_num]["symbols"]
        symbol = symbols[selected_index]
        model_prediction = symbol.get("original_class", symbol["class"])
        manual_correction = corrected_class.strip()

        symbol["original_class"] = model_prediction
        symbol["class"] = manual_correction
        symbol["manually_verified"] = True
        symbol["correction_source"] = "sql"

        document_id = app_state.get("document_id")
        tag = symbol.get("tag")
        if tag and document_id:
            save_correction(document_id, tag, model_prediction, manual_correction, symbol["bbox"])
            # Propagate to the SAME tag on OTHER pages only (a genuinely
            # unique tag can legitimately reappear across pages via
            # off-page references). Deliberately NOT propagating to other
            # symbols on this same page — two different symbols on one
            # page can share a nearby descriptive label (e.g. "Instrument
            # Air Supply") without being the same instrument, and this
            # loop used to relabel both just because the tag text matched.
            for page_num_iter, page_symbols in app_state["page_data"].items():
                if page_num_iter == page_num:
                    continue
                for s in page_symbols["symbols"]:
                    if s.get("tag") == tag:
                        s["original_class"] = s.get("original_class", s["class"])
                        s["class"] = manual_correction
                        s["manually_verified"] = True
                        s["correction_source"] = "sql"
            message = f"Updated class from '{model_prediction}' to '{manual_correction}'. Saved correction for tag '{tag}'."
        else:
            message = (
                f"Updated class from '{model_prediction}' to '{manual_correction}'. "
                "This symbol has no stable tag, so the correction is not persisted."
            )

        app_state.pop("manual_selected_index", None)

        return message, update_manual_verification_view(page_choice)
    except Exception as e:
        print(traceback.format_exc())
        return f"ERROR: {str(e)}", update_manual_verification_view(page_choice)


def remove_saved_correction(page_choice):
    try:
        page_num = app_state.get("manual_active_page_num")
        selected_index = app_state.get("manual_selected_index")

        if page_num is None or selected_index is None:
            return "No existing symbol selected. Click a detected symbol first.", update_manual_verification_view(page_choice)

        symbols = app_state["page_data"][page_num]["symbols"]
        symbol = symbols[selected_index]
        tag = symbol.get("tag")
        document_id = app_state.get("document_id")

        if not tag or not document_id:
            return "This symbol has no stable tag/document, so no saved correction can be removed.", update_manual_verification_view(page_choice)

        remove_correction(document_id, tag, symbol["bbox"])

        restored_count = 0
        # Restore this exact symbol first...
        if symbol.get("original_class") is not None:
            symbol["class"] = symbol["original_class"]
            symbol["manually_verified"] = False
            symbol.pop("correction_source", None)
            restored_count += 1

        # ...then the same tag on OTHER pages only — matching the same
        # same-page-collision reasoning as apply_manual_class_correction
        # above: a shared label on this page doesn't mean shared identity.
        for page_num_iter, page_symbols in app_state["page_data"].items():
            if page_num_iter == page_num:
                continue
            for s in page_symbols["symbols"]:
                if s.get("tag") == tag and s.get("original_class") is not None:
                    s["class"] = s["original_class"]
                    s["manually_verified"] = False
                    s.pop("correction_source", None)
                    restored_count += 1

        app_state.pop("manual_selected_index", None)

        if restored_count > 0:
            message = f"Removed saved correction for '{tag}' and restored the YOLO prediction on {restored_count} symbol(s)."
        else:
            message = f"Removed saved correction for '{tag}', but no in-memory symbols were restored."

        return message, update_manual_verification_view(page_choice)
    except Exception as e:
        print(traceback.format_exc())
        return f"ERROR: {str(e)}", update_manual_verification_view(page_choice)


def view_saved_changes():
    """Reads corrections.db for the currently-loaded document and returns
    two tables: every saved class correction, and every saved manual
    addition. Purely a read — doesn't touch app_state's in-memory symbols
    at all, so it's safe to refresh anytime."""
    correction_cols = ["tag", "position_bucket", "model_prediction",
                        "manual_correction_done_earlier", "created_at", "updated_at"]
    addition_cols = ["tag", "class", "page_num"]

    document_id = app_state.get("document_id")
    if not document_id:
        return pd.DataFrame(columns=correction_cols), pd.DataFrame(columns=addition_cols)

    corrections = list_corrections(document_id)
    corr_df = pd.DataFrame([
        {
            "tag": c["tag"],
            "position_bucket": c["position_bucket"],
            "model_prediction": c["model_prediction"],
            "manual_correction_done_earlier": c["manual_correction_done_earlier"],
            "created_at": c["created_at"],
            "updated_at": c["updated_at"],
        }
        for c in corrections
    ]) if corrections else pd.DataFrame(columns=correction_cols)

    additions = get_additions_for_document(document_id)
    add_df = pd.DataFrame([
        {"tag": a["tag"], "class": a["class"], "page_num": a["page_num"]}
        for a in additions
    ]) if additions else pd.DataFrame(columns=addition_cols)

    return corr_df, add_df


def delete_manual_symbol(page_choice):
    try:
        page_num = app_state.get("manual_active_page_num")
        selected_index = app_state.get("manual_selected_index")

        if page_num is None or selected_index is None:
            return "No symbol selected for deletion. Click a symbol first.", update_manual_verification_view(page_choice)

        symbols = app_state["page_data"][page_num]["symbols"]
        removed = symbols.pop(selected_index)

        # If this was a manually-added symbol saved earlier, remove its
        # persisted record too — otherwise it would silently reappear the
        # next time this document is uploaded.
        document_id = app_state.get("document_id")
        if removed.get("manually_added") and removed.get("tag") and document_id:
            remove_addition(document_id, removed["tag"], removed["bbox"])

        app_state.pop("manual_selected_index", None)

        return (
            f"Removed '{removed['class']}'.",
            update_manual_verification_view(page_choice)
        )
    except Exception as e:
        print(traceback.format_exc())
        return f"ERROR: {str(e)}", update_manual_verification_view(page_choice)


def add_manual_symbol(page_choice, symbol_class):
    try:
        if not symbol_class or not symbol_class.strip():
            return "Please enter a symbol class name.", update_manual_verification_view(page_choice)

        bbox = app_state.get("manual_new_bbox")
        page_num = app_state.get("manual_active_page_num")

        if bbox is None or page_num is None:
            return "No bounding box set. Click two corners around the missed symbol first.", update_manual_verification_view(page_choice)

        symbol_class = symbol_class.strip()

        # Try to find a nearby OCR-read tag for this new box, the same way
        # the initial detection pass does (see tag_matching.assign_tags) —
        # without a tag there's nothing stable to persist this addition
        # against, so it would silently vanish on the next upload.
        text_detections = app_state["page_data"][page_num].get("text", [])
        tag = None
        best_dist = float("inf")
        new_center = bbox_center(bbox)
        for t in text_detections:
            d = np.linalg.norm(np.array(new_center) - np.array(bbox_center(t["bbox"])))
            if d < best_dist and d <= 80:
                best_dist, tag = d, t["text"]

        manual_detection = {
            "class": symbol_class,
            "confidence": 1.0,
            "bbox": bbox,
            "tag": tag,
            "manually_added": True,
            "manually_verified": True,
        }

        app_state["page_data"][page_num]["symbols"].append(manual_detection)
        app_state.pop("manual_new_bbox", None)

        document_id = app_state.get("document_id")
        if tag and document_id:
            save_addition(document_id, tag, symbol_class, bbox, page_num)
            persistence_note = f" Saved — will auto-apply to '{tag}' on future runs of this document."
        else:
            persistence_note = (" No tag was found near this box, so it only applies to the current "
                                 "session — there's nothing to key a persistent lookup on.")

        return (
            f"Added new symbol: '{symbol_class}'.{persistence_note}",
            update_manual_verification_view(page_choice)
        )
    except Exception as e:
        print(traceback.format_exc())
        return f"ERROR: {str(e)}", update_manual_verification_view(page_choice)


def update_manual_active_page(page_choice):
    app_state.pop("manual_first_point", None)
    app_state.pop("manual_selected_index", None)
    app_state.pop("manual_new_bbox", None)
    app_state.pop("manual_active_page_num", None)
    return update_manual_verification_view(page_choice)


def refresh_page_derived_data(page_choice):
    """Recomputes lines, arrows, and connectivity after any manual edit,
    using the SAME masked-mask logic as initial page processing —
    keeps Full Detection, Asset Hierarchy, and manual edits all in sync."""
    if not page_choice or not app_state["page_paths"]:
        return
    page_num = int(page_choice.split(" ")[1])
    page_path = app_state["page_paths"][page_num - 1]
    page_data = app_state["page_data"][page_num]

    lines, arrows, connectivity = recompute_lines_and_connectivity(
        page_path, page_data["symbols"], page_data["text"]
    )
    page_data["lines"] = lines
    page_data["arrows"] = arrows
    page_data["connectivity"] = connectivity


# ============================================================
# ASSET HIERARCHY
# ============================================================

def update_hierarchy_view(page_choice):
    empty_cols = ["Symbol ID", "Method", "Symbol Type", "Associated Text", "Connected Symbols"]
    if not page_choice or not app_state["page_paths"]:
        return pd.DataFrame(columns=empty_cols)
    page_num = int(page_choice.split(" ")[1])
    page_data = app_state["page_data"][page_num]
    connectivity = page_data.get("connectivity") or {}
    return build_asset_hierarchy_table(page_data["symbols"], connectivity)


def update_hierarchy_image(page_choice):
    if not page_choice or not app_state["page_paths"]:
        return None
    page_num = int(page_choice.split(" ")[1])
    page_path = app_state["page_paths"][page_num - 1]
    page_data = app_state["page_data"][page_num]
    return draw_symbols_only(page_path, page_data)


def handle_hierarchy_click(page_choice, evt: gr.SelectData):
    try:
        x, y = evt.index
        page_num = int(page_choice.split(" ")[1])
        page_path = app_state["page_paths"][page_num - 1]
        page_data = app_state["page_data"][page_num]
        symbols = page_data["symbols"]

        clicked = find_symbol_at_click(symbols, x, y)
        if clicked is None:
            return draw_symbols_only(page_path, page_data), "No symbol found at that click."

        selected_index = symbols.index(clicked)

        connectivity_map = page_data.get("connectivity") or {}
        if not connectivity_map:
            _, _, connectivity_map = recompute_lines_and_connectivity(page_path, symbols, page_data["text"])
            page_data["connectivity"] = connectivity_map

        img = draw_connectivity_click_view(page_path, symbols, connectivity_map, selected_index)

        connected_ids = connectivity_map.get(selected_index, {})
        connected_desc = [f"{i}: {symbols[i]['class']}" for i in connected_ids.keys()]
        summary = (
            f"Selected: {clicked['class']} (ID {selected_index})\nConnected to:\n" + "\n".join(connected_desc)
            if connected_desc else
            f"Selected: {clicked['class']} — no connections found."
        )
        return img, summary
    except Exception as e:
        print(traceback.format_exc())
        try:
            page_num = int(page_choice.split(" ")[1])
            page_path = app_state["page_paths"][page_num - 1]
            page_data = app_state["page_data"][page_num]
            fallback_img = draw_symbols_only(page_path, page_data)
        except Exception:
            fallback_img = None
        return fallback_img, f"ERROR: {str(e)}"


# ============================================================
# GRADIO INTERFACE BUILD
# ============================================================

with gr.Blocks(title="P&ID Detection Suite") as demo:
    gr.Markdown("# P&ID Detection & Symbol Matcher")

    with gr.Row():
        file_input = gr.File(label="Upload P&ID (PDF or image)", type="filepath", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
        upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("Full Detection"):
            with gr.Row():
                page_dropdown_1 = gr.Dropdown(label="Page", choices=[])
                view_choice = gr.Radio(["Symbols only", "Text only", "Lines only", "Lines + Arrows", "All combined"],
                                        value="All combined", label="View")
            detection_output = gr.Image(label="Detection result")
            detection_summary_table = gr.Dataframe(label="Symbol counts on this page", headers=["Symbol Class", "Count"])

            page_dropdown_1.change(update_detection_view, inputs=[page_dropdown_1, view_choice],
                                    outputs=[detection_output, detection_summary_table])
            view_choice.change(update_detection_view, inputs=[page_dropdown_1, view_choice],
                                outputs=[detection_output, detection_summary_table])

        with gr.Tab("Symbol Matcher"):
            page_dropdown_2 = gr.Dropdown(label="Page to select a symbol from", choices=[])
            gr.Markdown("Click directly on any detected symbol (green box) to find all matching symbols across every page.")
            matcher_image = gr.Image(label="Click a symbol", type="numpy", interactive=False)
            reset_button = gr.Button("Select New Symbol (Reset)")
            match_gallery = gr.Gallery(label="Matches found (all pages)", columns=2)
            match_summary = gr.Textbox(label="Summary", lines=10)
            matcher_summary_table = gr.Dataframe(label="Symbol counts on this page", headers=["Symbol Class", "Count"])

            page_dropdown_2.change(update_matcher_view, inputs=[page_dropdown_2],
                                    outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table])
            matcher_image.select(handle_symbol_click, inputs=[page_dropdown_2],
                                  outputs=[match_gallery, match_summary])
            reset_button.click(handle_reset, inputs=[page_dropdown_2],
                                outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table])

        with gr.Tab("Manual Verification"):
            gr.Markdown(
                """
                ### Manual Symbol Verification & Editing

                **Colors:** green = model detection · blue = manually corrected · orange = manually added · red = currently selected

                * **Correct / Delete Symbol:** click a detected box, then rename or delete it.
                * **Add Missing Symbol:** click two opposite corners around a missed symbol, enter its class, then add it.

                Edits here automatically recompute lines, arrows, and connectivity for this page.
                """
            )

            with gr.Row():
                page_dropdown_3 = gr.Dropdown(label="Page", choices=[])
                manual_mode = gr.Radio(
                    ["Correct / Delete Symbol", "Add Missing Symbol"],
                    value="Correct / Delete Symbol",
                    label="Verification Mode"
                )

            manual_image = gr.Image(
                label="Click symbols or bounding-box corners",
                type="numpy",
                interactive=False,
            )

            manual_class_input = gr.Textbox(
                label="Symbol Class Name",
                placeholder="Example: butterfly_valve"
            )

            with gr.Row():
                apply_correction_button = gr.Button("Apply Class Correction", variant="primary")
                delete_symbol_button = gr.Button("Delete Selected Symbol", variant="stop")
                add_symbol_button = gr.Button("Add Missing Symbol")
                remove_saved_correction_button = gr.Button("Restore YOLO Prediction", variant="secondary")

            manual_status = gr.Textbox(
                label="Manual Verification Log",
                lines=4,
                interactive=False
            )

            with gr.Accordion("Saved Corrections & Additions for this document", open=False):
                view_saved_changes_button = gr.Button("Refresh")
                saved_corrections_table = gr.Dataframe(label="Saved Class Corrections")
                saved_additions_table = gr.Dataframe(label="Saved Manual Additions")

            view_saved_changes_button.click(
                view_saved_changes,
                inputs=[],
                outputs=[saved_corrections_table, saved_additions_table]
            )

            page_dropdown_3.change(
                update_manual_active_page,
                inputs=[page_dropdown_3],
                outputs=[manual_image]
            )

            manual_image.select(
                manual_verification_click,
                inputs=[page_dropdown_3, manual_mode],
                outputs=[manual_status, manual_image]
            )

            apply_correction_button.click(
                apply_manual_class_correction,
                inputs=[page_dropdown_3, manual_class_input],
                outputs=[manual_status, manual_image]
            ).then(
                refresh_page_derived_data, inputs=[page_dropdown_3], outputs=[]
            ).then(
                update_detection_view, inputs=[page_dropdown_1, view_choice], outputs=[detection_output, detection_summary_table]
            ).then(
                update_matcher_view, inputs=[page_dropdown_2], outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table]
            )

            delete_symbol_button.click(
                delete_manual_symbol,
                inputs=[page_dropdown_3],
                outputs=[manual_status, manual_image]
            ).then(
                refresh_page_derived_data, inputs=[page_dropdown_3], outputs=[]
            ).then(
                update_detection_view, inputs=[page_dropdown_1, view_choice], outputs=[detection_output, detection_summary_table]
            ).then(
                update_matcher_view, inputs=[page_dropdown_2], outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table]
            )

            add_symbol_button.click(
                add_manual_symbol,
                inputs=[page_dropdown_3, manual_class_input],
                outputs=[manual_status, manual_image]
            ).then(
                refresh_page_derived_data, inputs=[page_dropdown_3], outputs=[]
            ).then(
                update_detection_view, inputs=[page_dropdown_1, view_choice], outputs=[detection_output, detection_summary_table]
            ).then(
                update_matcher_view, inputs=[page_dropdown_2], outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table]
            )

            remove_saved_correction_button.click(
                remove_saved_correction,
                inputs=[page_dropdown_3],
                outputs=[manual_status, manual_image]
            ).then(
                refresh_page_derived_data, inputs=[page_dropdown_3], outputs=[]
            ).then(
                update_detection_view, inputs=[page_dropdown_1, view_choice], outputs=[detection_output, detection_summary_table]
            ).then(
                update_matcher_view, inputs=[page_dropdown_2], outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table]
            )

        with gr.Tab("Asset Hierarchy"):
            gr.Markdown("Click a symbol below to see what it's connected to (red = selected, green = connected, cyan = traced line path).")
            page_dropdown_4 = gr.Dropdown(label="Page", choices=[])
            hierarchy_image = gr.Image(label="Click a symbol", type="numpy", interactive=False)
            hierarchy_click_summary = gr.Textbox(label="Connections", lines=6)
            hierarchy_table = gr.Dataframe(label="Asset Hierarchy Table")

            page_dropdown_4.change(update_hierarchy_view, inputs=[page_dropdown_4], outputs=[hierarchy_table])
            page_dropdown_4.change(update_hierarchy_image, inputs=[page_dropdown_4], outputs=[hierarchy_image])
            hierarchy_image.select(handle_hierarchy_click, inputs=[page_dropdown_4], outputs=[hierarchy_image, hierarchy_click_summary])

        with gr.Tab("Ask the Diagram"):
            gr.Markdown(
                """
                ### Ask questions about this page in plain English
                Answers use only what's been detected on this page - the LLM
                can't see anything the pipeline missed. Ollama runs locally
                and is free; OpenRouter is a paid hosted API - see the top
                of `llm_query.py` for setup of either.
                """
            )
            page_dropdown_5 = gr.Dropdown(label="Page", choices=[])
            backend_dropdown = gr.Dropdown(
                label="Answer using",
                choices=["Ollama (local, free)", "OpenRouter (hosted, paid)"],
                value="Ollama (local, free)",
            )
            diagram_chatbot = gr.Chatbot(label="Diagram Q&A", height=420)
            diagram_question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is FT-101 connected to?",
            )
            diagram_ask_button = gr.Button("Ask", variant="primary")

            page_dropdown_5.change(handle_diagram_chat_page_change, inputs=[page_dropdown_5], outputs=[diagram_chatbot])
            diagram_ask_button.click(
                handle_diagram_chat,
                inputs=[page_dropdown_5, diagram_question_box, diagram_chatbot, backend_dropdown],
                outputs=[diagram_chatbot, diagram_question_box],
            )
            diagram_question_box.submit(
                handle_diagram_chat,
                inputs=[page_dropdown_5, diagram_question_box, diagram_chatbot, backend_dropdown],
                outputs=[diagram_chatbot, diagram_question_box],
            )

    file_input.upload(
        handle_upload,
        inputs=[file_input],
        outputs=[page_dropdown_1, page_dropdown_2, page_dropdown_3, page_dropdown_4, page_dropdown_5, upload_status]
    ).then(
        update_detection_view, inputs=[page_dropdown_1, view_choice], outputs=[detection_output, detection_summary_table]
    ).then(
        update_matcher_view, inputs=[page_dropdown_2], outputs=[matcher_image, match_gallery, match_summary, matcher_summary_table]
    ).then(
        update_manual_verification_view, inputs=[page_dropdown_3], outputs=[manual_image]
    ).then(
        update_hierarchy_view, inputs=[page_dropdown_4], outputs=[hierarchy_table]
    ).then(
        update_hierarchy_image, inputs=[page_dropdown_4], outputs=[hierarchy_image]
    )

if __name__ == "__main__":
  demo.launch(share=True, debug=True) 