"""
Persistent human-verification layer, sitting between YOLO's raw output and
the rest of the pipeline (lines/connectivity/arrows/QA).

Design, matching the flow you sketched:

    YOLO best.pt -> initial detection -> check SQL corrections ->
    apply corrected class (if any) / keep YOLO result -> corrected result
    -> lines/connectivity -> P&ID graph

best.pt is never touched here. A correction is keyed by
(document, tag, position_bucket) — tag rather than page number, because
the same instrument tag can legitimately reappear across multiple pages
of one P&ID set (off-page references), and a human's "XV-101 is actually
a gate valve" verification should apply everywhere that tag shows up in
that document.

position_bucket exists because "tag" here really means "whatever OCR text
was nearest to the symbol" — and that isn't always a unique instrument
tag. A descriptive label like "Instrument Air Supply" can legitimately
sit next to two different valves in the same diagram. Without a position
component, correcting the second one would silently overwrite the first
one's row (same document+tag primary key), which is exactly the bug this
was built to prevent: two symbols, same nearby label, only one survived a
re-upload. position_bucket rounds each symbol's bbox center to a coarse
grid, so two symbols sharing a tag text but sitting in different places
on the page get separate rows, while re-processing the same document
(same rendering, same coordinates) still lands in the same bucket and
correctly reapplies the correction.

Symbols with no OCR-matched tag (tag is None) can't be looked up this way
— there's nothing stable to key on — so they're intentionally NOT
persisted here. The UI should say so when a correction is applied to an
untagged symbol (see app.py's apply_manual_class_correction).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "corrections.db"
FALLBACK_DB_DIR = Path.home() / ".pid_symbol_detector"

POSITION_BUCKET_SIZE = 40  # pixels. Coarse enough to tolerate tiny
                           # rendering jitter between runs of the same
                           # document, fine enough to separate two
                           # distinct symbols that happen to share a
                           # nearby label.


def _get_db_path():
    if DB_PATH.parent.exists() and os.access(DB_PATH.parent, os.W_OK):
        return DB_PATH

    FALLBACK_DB_DIR.mkdir(parents=True, exist_ok=True)
    return FALLBACK_DB_DIR / "corrections.db"


def _position_bucket(bbox, bucket_size=POSITION_BUCKET_SIZE):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return f"{int(cx // bucket_size)}_{int(cy // bucket_size)}"


def _table_has_column(conn, table_name, column_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def _ensure_corrections_schema(conn):
    if not _table_has_column(conn, "corrections", "position_bucket"):
        rows = conn.execute("PRAGMA table_info(corrections)").fetchall()
        old_columns = {row[1] for row in rows}

        conn.execute("ALTER TABLE corrections RENAME TO corrections_old")
        conn.execute("""
            CREATE TABLE corrections (
                document TEXT NOT NULL,
                tag TEXT NOT NULL,
                position_bucket TEXT NOT NULL,
                model_prediction TEXT,
                manual_correction_done_earlier TEXT NOT NULL,
                correction_verified INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (document, tag, position_bucket)
            )
        """)

        model_prediction_column = (
            "model_prediction"
            if "model_prediction" in old_columns
            else "original_class"
            if "original_class" in old_columns
            else "NULL"
        )
        manual_correction_column = (
            "manual_correction_done_earlier"
            if "manual_correction_done_earlier" in old_columns
            else "corrected_class"
            if "corrected_class" in old_columns
            else "''"
        )
        correction_verified_column = (
            "correction_verified"
            if "correction_verified" in old_columns
            else "1"
        )
        created_at_column = (
            "created_at"
            if "created_at" in old_columns
            else "updated_at"
            if "updated_at" in old_columns
            else "'1970-01-01T00:00:00Z'"
        )
        updated_at_column = (
            "updated_at"
            if "updated_at" in old_columns
            else created_at_column
        )

        conn.execute(f"INSERT INTO corrections (document, tag, position_bucket, model_prediction, manual_correction_done_earlier, correction_verified, created_at, updated_at) SELECT document, tag, '', {model_prediction_column}, {manual_correction_column}, {correction_verified_column}, {created_at_column}, {updated_at_column} FROM corrections_old")
        conn.execute("DROP TABLE IF EXISTS corrections_old")


def _get_connection():
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            document TEXT NOT NULL,
            tag TEXT NOT NULL,
            position_bucket TEXT NOT NULL,
            model_prediction TEXT,
            manual_correction_done_earlier TEXT NOT NULL,
            correction_verified INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (document, tag, position_bucket)
        )
    """)
    _ensure_corrections_schema(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manual_additions (
            document TEXT NOT NULL,
            tag TEXT NOT NULL,
            position_bucket TEXT NOT NULL,
            class TEXT NOT NULL,
            bbox TEXT NOT NULL,
            page_num INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (document, tag, position_bucket)
        )
    """)
    return conn


def save_correction(document, tag, model_prediction, manual_correction_done_earlier, bbox):
    """Upserts a correction, keyed by (document, tag, position of the
    symbol being corrected) — see module docstring for why position is
    part of the key.

    The model prediction is preserved forever for this document/tag/
    position. Later manual re-verifications update only the human
    correction, verified flag, and updated_at, while keeping
    model_prediction and created_at intact.
    """
    if not document or not tag:
        return  # nothing stable to key on — caller should have checked this

    bucket = _position_bucket(bbox)
    now = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO corrections (
                document, tag, position_bucket, model_prediction, manual_correction_done_earlier,
                correction_verified, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document, tag, position_bucket) DO UPDATE SET
                manual_correction_done_earlier = excluded.manual_correction_done_earlier,
                correction_verified = excluded.correction_verified,
                updated_at = excluded.updated_at
        """, (
            document,
            tag,
            bucket,
            model_prediction,
            manual_correction_done_earlier,
            1,
            now,
            now,
        ))


def remove_correction(document, tag, bbox):
    """Deletes a previously saved correction so the original YOLO
    prediction can be restored. bbox is required now too, so removing
    one of two same-tag symbols doesn't accidentally delete the other's
    correction."""
    if not document or not tag:
        return

    bucket = _position_bucket(bbox)
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM corrections WHERE document = ? AND tag = ? AND position_bucket = ?",
            (document, tag, bucket),
        )


def get_corrections_for_document(document):
    """Returns {(tag, position_bucket): corrected_class} for every
    verified correction on this document. Called once per page-processing
    run, not per symbol — cheap enough to just re-query."""
    if not document:
        return {}

    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT tag, position_bucket, manual_correction_done_earlier FROM corrections WHERE document = ?",
            (document,),
        ).fetchall()
    return {(tag, bucket): corrected for tag, bucket, corrected in rows}


def list_corrections(document):
    """Full correction log for a document (for display/audit in the UI),
    newest first."""
    if not document:
        return []

    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT tag, position_bucket, model_prediction, manual_correction_done_earlier, correction_verified, created_at, updated_at
            FROM corrections WHERE document = ?
            ORDER BY updated_at DESC
        """, (document,)).fetchall()

    return [
        {
            "tag": tag,
            "position_bucket": bucket,
            "model_prediction": model_prediction,
            "manual_correction_done_earlier": manual_correction_done_earlier,
            "correction_verified": bool(correction_verified),
            "created_at": created_at,
            "updated_at": updated_at,
        }
        for tag, bucket, model_prediction, manual_correction_done_earlier, correction_verified, created_at, updated_at in rows
    ]


def apply_corrections(symbols, document):
    """Overrides symbol['class'] wherever a verified correction exists for
    its (tag, position). Leaves untagged symbols and symbols with no
    matching correction untouched. Marks corrected symbols so the UI can
    visually distinguish "YOLO said this" from "a human verified this"."""
    corrections = get_corrections_for_document(document)
    if not corrections:
        return symbols

    for s in symbols:
        tag = s.get("tag")
        if not tag:
            continue
        key = (tag, _position_bucket(s["bbox"]))
        if key in corrections:
            s["original_class"] = s["class"]
            s["class"] = corrections[key]
            s["manually_verified"] = True
            s["correction_source"] = "sql"

    return symbols


def save_addition(document, tag, class_name, bbox, page_num):
    """Persists a symbol a human added that YOLO never detected at all —
    distinct from save_correction, which overrides an *existing*
    detection's class. Keyed the same way as corrections: (document, tag,
    position) — see module docstring for why position is part of the key.

    bbox is stored as JSON so the symbol can be redrawn at the same
    location on a future run. This assumes the document renders at the
    same resolution each time (same PDF, same conversion settings) — if
    that ever changes, the restored bbox could land in the wrong place.
    """
    if not document or not tag:
        return  # nothing stable to key on — caller should have checked this

    bucket = _position_bucket(bbox)
    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO manual_additions (document, tag, position_bucket, class, bbox, page_num, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document, tag, position_bucket) DO UPDATE SET
                class = excluded.class,
                bbox = excluded.bbox,
                page_num = excluded.page_num
        """, (document, tag, bucket, class_name, json.dumps(bbox), page_num, datetime.now(timezone.utc).isoformat()))


def remove_addition(document, tag, bbox):
    """Deletes a previously saved manual addition — so deleting an added
    symbol in the UI doesn't just remove it for this session while it
    quietly reappears on the next upload. bbox required for the same
    reason as remove_correction: don't delete a sibling with the same tag."""
    if not document or not tag:
        return

    bucket = _position_bucket(bbox)
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM manual_additions WHERE document = ? AND tag = ? AND position_bucket = ?",
            (document, tag, bucket),
        )


def get_additions_for_document(document, page_num=None):
    """Returns saved manual additions for this document, optionally
    filtered to one page. Each entry: {tag, class, bbox, page_num}."""
    if not document:
        return []

    query = "SELECT tag, class, bbox, page_num FROM manual_additions WHERE document = ?"
    params = [document]
    if page_num is not None:
        query += " AND page_num = ?"
        params.append(page_num)

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {"tag": tag, "class": class_name, "bbox": json.loads(bbox), "page_num": pg}
        for tag, class_name, bbox, pg in rows
    ]


def apply_additions(symbols, document, page_num):
    """Injects previously saved manual additions for this page as new
    symbol entries — these have no YOLO detection to override, so unlike
    apply_corrections this adds rows rather than modifying existing ones.
    Skips a saved addition if a symbol already exists at essentially the
    same (tag, position) — e.g. the model has since learned to detect it
    — to avoid a visible duplicate box. Position is part of the check
    (not just tag) for the same reason it's part of the DB key: two
    genuinely different additions can share a nearby label."""
    additions = get_additions_for_document(document, page_num=page_num)
    if not additions:
        return symbols

    existing_keys = {
        (s.get("tag"), _position_bucket(s["bbox"]))
        for s in symbols if s.get("tag")
    }
    for a in additions:
        key = (a["tag"], _position_bucket(a["bbox"]))
        if key in existing_keys:
            continue
        symbols.append({
            "class": a["class"],
            "confidence": 1.0,
            "bbox": a["bbox"],
            "tag": a["tag"],
            "manually_added": True,
            "manually_verified": True,
            "correction_source": "sql",
        })

    return symbols
