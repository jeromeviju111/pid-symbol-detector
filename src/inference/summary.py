import pandas as pd
from collections import Counter


def build_symbol_summary_table(detections):
    class_counts = Counter(d["class"] for d in detections)
    df = pd.DataFrame(
        [(cls, count) for cls, count in class_counts.most_common()],
        columns=["Symbol Class", "Count"]
    )
    return df
