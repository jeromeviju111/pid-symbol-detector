import pandas as pd


def build_asset_hierarchy_table(detections, connectivity_map=None):
    rows = []
    conn_map = connectivity_map or {}
    for i, det in enumerate(detections):
        connected_ids = conn_map.get(i, {})
        rows.append({
            "Symbol ID": i,
            "Method": "Manually-labeled" if det.get("manually_verified") else "Detected",
            "Symbol Type": det["class"],
            "Associated Text": det.get("tag") or "",
            "Connected Symbols": ", ".join(str(c) for c in connected_ids.keys()),
        })
    return pd.DataFrame(rows)