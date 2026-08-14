"""
Turns the raw pipeline output (page_data: symbols/lines/arrows/connectivity)
into a compact, LLM-readable graph. This is deliberately NOT the full
detection data - pixel coordinates and confidence scores are stripped out
because they add nothing for a Q&A LLM and just burn context budget.

v1 note: edges are undirected. The connectivity_map traces which symbols are
linked, but doesn't yet attach a specific arrow to a specific edge, so flow
direction isn't encoded here. Once arrow detection is validated on your
diagrams, this can be extended to match each arrow's position against the
connectivity path it sits on and mark that edge as directional.
"""

from collections import Counter


def build_diagram_graph(page_data, page_label="Page 1"):
    symbols = page_data.get("symbols", [])
    connectivity = page_data.get("connectivity", {}) or {}

    nodes = []
    for i, s in enumerate(symbols):
        nodes.append({
            "id": i,
            "tag": s.get("tag") or f"UNTAGGED-{i}",
            "type": s["class"],
        })

    edges = []
    seen = set()
    for src_id, connections in connectivity.items():
        for dst_id in connections.keys():
            key = tuple(sorted((int(src_id), int(dst_id))))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": key[0], "to": key[1]})

    counts = dict(Counter(n["type"] for n in nodes))

    return {"page": page_label, "nodes": nodes, "edges": edges, "counts": counts}


def merge_page_graphs(graphs_by_page):
    """Combine multiple per-page graphs (dict: page_label -> page_data) into
    one multi-page graph, keeping node ids unique across pages."""
    all_nodes, all_edges = [], []
    offset = 0
    for page_label, page_data in graphs_by_page.items():
        g = build_diagram_graph(page_data, page_label=page_label)
        for n in g["nodes"]:
            all_nodes.append({**n, "id": n["id"] + offset, "page": page_label})
        for e in g["edges"]:
            all_edges.append({"from": e["from"] + offset, "to": e["to"] + offset})
        offset += len(g["nodes"])
    counts = dict(Counter(n["type"] for n in all_nodes))
    return {"nodes": all_nodes, "edges": all_edges, "counts": counts}


def graph_to_text(graph):
    """Serialize the graph into plain text for LLM context. Grouped by tag
    so the model can answer 'what is X connected to' without having to
    cross-reference two separate lists.

    Counts are computed here in Python and handed to the model as exact
    numbers, rather than making the model count list entries itself -
    counting items in a long text list is a well-known LLM weak spot
    regardless of model size. "How many ball valves" should be answered
    by reading one line here, not by counting 22 bullet points below it.
    """
    tag_by_id = {n["id"]: n["tag"] for n in graph["nodes"]}
    page_by_id = {n["id"]: n.get("page") for n in graph["nodes"]}

    adjacency = {n["id"]: [] for n in graph["nodes"]}
    for e in graph["edges"]:
        adjacency[e["from"]].append(e["to"])
        adjacency[e["to"]].append(e["from"])

    lines = ["P&ID Diagram Extraction", ""]

    counts = graph.get("counts") or {}
    if counts:
        lines.append("Exact symbol counts by type (authoritative - use these numbers "
                      "directly for any 'how many' question, do not recount the list below):")
        for symbol_type, count in sorted(counts.items()):
            lines.append(f"- {symbol_type}: {count}")
        lines.append(f"Total symbols: {sum(counts.values())}")
        lines.append("")

    lines.append("Individual symbols and connections:")
    for n in graph["nodes"]:
        neighbors = adjacency.get(n["id"], [])
        neighbor_desc = ", ".join(tag_by_id.get(nb, str(nb)) for nb in neighbors) or "none detected"
        page_str = f" (page: {page_by_id[n['id']]})" if page_by_id.get(n["id"]) else ""
        lines.append(f"- {n['tag']} [{n['type']}]{page_str} -> connected to: {neighbor_desc}")

    return "\n".join(lines)


def try_answer_count_question(question, counts):
    """Direct, code-computed answer for 'how many X' style questions -
    bypasses the LLM entirely for this pattern. Counting items in a long
    text list is unreliable for LLMs regardless of size or how the system
    prompt is worded (they can still ignore an instruction and recount the
    list themselves); this makes the count question actually deterministic
    instead of just "more likely correct".

    Returns None if the question doesn't look like a count question, or the
    symbol type isn't recognized - callers should fall through to the LLM
    as normal in that case.
    """
    import re

    if not question or not counts:
        return None

    q = question.strip()
    m = re.search(r"how many\s+(.+?)\??$", q, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:count|number|total) of\s+(.+?)\??$", q, re.IGNORECASE)
    if not m:
        return None

    phrase = m.group(1).strip().lower()
    # Strip trailing filler clauses regardless of order/combination, rather
    # than trying to enumerate every possible phrasing in one regex.
    trailing_fillers = [
        r"\s+are there\b.*", r"\s+exist\b.*", r"\s+do you see\b.*",
        r"\s+were detected\b.*", r"\s+does (?:it|this|the diagram) have\b.*",
        r"\s+symbols?\b.*",
    ]
    for pattern in trailing_fillers:
        phrase = re.sub(pattern, "", phrase).strip()
    if not phrase:
        return None

    candidates = {phrase, phrase.replace(" ", "_"), phrase.replace("-", "_")}
    for c in list(candidates):
        if c.endswith("s") and len(c) > 1:
            candidates.add(c[:-1])  # naive plural -> singular (valves -> valve)

    counts_lower = {k.lower(): (k, v) for k, v in counts.items()}
    for c in candidates:
        if c in counts_lower:
            real_name, n = counts_lower[c]
            return f"There are {n} {real_name} symbol(s) detected on this page."

    return None
