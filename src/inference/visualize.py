import cv2


def draw_symbols_only(page_path, page_data):
    img = cv2.imread(page_path)
    for d in page_data["symbols"]:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d["class"] + (f" [{d['tag']}]" if d.get("tag") else "")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_text_only(page_path, page_data):
    img = cv2.imread(page_path)
    for t in page_data["text"]:
        x1, y1, x2, y2 = map(int, t["bbox"])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(img, t["text"], (int(x1), max(int(y1)-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_lines_only(page_path, page_data):
    img = cv2.imread(page_path)
    for ll in page_data["lines"]:
        x1, y1, x2, y2 = ll["line"]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        if ll["line_id"]:
            cv2.putText(img, ll["line_id"], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_lines_and_arrows(page_path, page_data):
    img = cv2.imread(page_path)
    for ll in page_data["lines"]:
        x1, y1, x2, y2 = ll["line"]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    for a in page_data["arrows"]:
        x1, y1, x2, y2 = map(int, a["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, a["direction"], (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_combined(page_path, page_data):
    img = cv2.imread(page_path)
    for ll in page_data["lines"]:
        x1, y1, x2, y2 = ll["line"]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
    for d in page_data["symbols"]:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d["class"] + (f" [{d['tag']}]" if d.get("tag") else "")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 0), 1)
    for t in page_data["text"]:
        x1, y1, x2, y2 = map(int, t["bbox"])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_connectivity_click_view(page_path, symbols, connectivity_map, selected_index):
    img = cv2.imread(page_path)
    connections = connectivity_map.get(selected_index, {})

    for i, d in enumerate(symbols):
        x1, y1, x2, y2 = map(int, d["bbox"])
        if i == selected_index:
            color = (0, 0, 255)
        elif i in connections:
            color = (0, 255, 0)
        else:
            color = (180, 180, 180)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if i == selected_index or i in connections:
            cv2.putText(img, d["class"], (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # draw the actual traced path for each connection
    for connected_idx, path in connections.items():
        for j in range(len(path) - 1):
            cv2.line(img, path[j], path[j+1], (0, 255, 255), 2)  # cyan path line

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


VIEW_FUNCTIONS = {
    "Symbols only": draw_symbols_only,
    "Text only": draw_text_only,
    "Lines only": draw_lines_only,
    "Lines + Arrows": draw_lines_and_arrows,
    "All combined": draw_combined,
}


