import os
import yaml
import glob
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def load_yaml_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names, data


def count_gt_instances(data_yaml_path, split="val"):
    """Scan the YOLO label .txt files for the given split directly and count
    how many ground-truth boxes exist per class id. This is independent of
    anything ultralytics computes internally, so it tells you the ground
    truth about what's actually annotated in your val split."""
    _, data = load_yaml_names(data_yaml_path)
    base = Path(data_yaml_path).parent
    split_path = data.get(split)
    if split_path is None:
        raise ValueError(f"'{split}' key not found in data.yaml")

    images_dir = (base / split_path) if not os.path.isabs(split_path) else Path(split_path)
    labels_dir = Path(str(images_dir).replace("images", "labels"))

    counts = {}
    label_files = glob.glob(str(labels_dir / "*.txt"))
    for lf in label_files:
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts, len(label_files)


def compare_name_mappings(model_names, yaml_names):
    mismatches = []
    all_ids = sorted(set(model_names.keys()) | set(yaml_names.keys()))
    for i in all_ids:
        m = model_names.get(i, "<MISSING>")
        y = yaml_names.get(i, "<MISSING>")
        if m != y:
            mismatches.append((i, m, y))
    return mismatches


def evaluate_yolo_symbols(model_path, data_yaml_path, split="val", conf=0.001):
    print("=" * 60)
    print("STEP 1: Checking class-name alignment (model vs data.yaml)")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        return

    model = YOLO(model_path)
    model_names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    yaml_names, _ = load_yaml_names(data_yaml_path)

    mismatches = compare_name_mappings(model_names, yaml_names)
    if mismatches:
        print(f"\n*** WARNING: {len(mismatches)} class-id mismatches between the "
              f"model and this data.yaml. ***")
        print("This is the #1 cause of classes reading 0 P/R/mAP even though "
              "the model clearly fires on them during normal inference.\n")
        print(f"{'ID':<5}{'Model name':<30}{'data.yaml name':<30}")
        for i, m, y in mismatches:
            print(f"{i:<5}{m:<30}{y:<30}")
        print("\nIf this list is long: your data.yaml was very likely "
              "re-exported/re-versioned in Roboflow AFTER the model was "
              "trained, and the class order shifted underneath you. Point "
              "data.yaml at the exact dataset version the model was trained "
              "on, or reorder its 'names' list to match model.names exactly.\n")
    else:
        print("Model and data.yaml class names line up perfectly.\n")

    print("=" * 60)
    print("STEP 2: Counting ground-truth instances directly from label files")
    print("=" * 60)
    gt_counts, n_label_files = count_gt_instances(data_yaml_path, split=split)
    print(f"Scanned {n_label_files} label files in the '{split}' split.\n")

    print("=" * 60)
    print("STEP 3: Running YOLO validation")
    print("=" * 60)
    metrics = model.val(data=data_yaml_path, split=split, conf=conf)

    ap_indices = list(metrics.ap_class_index)
    per_class_metrics = {}
    for i, class_id in enumerate(ap_indices):
        per_class_metrics[int(class_id)] = {
            "Precision": round(float(metrics.box.p[i]), 4),
            "Recall": round(float(metrics.box.r[i]), 4),
            "mAP@50": round(float(metrics.box.ap50[i]), 4),
        }

    print("=" * 60)
    print("STEP 4: Full per-class report (every class, not just the ones")
    print("        ultralytics happened to include)")
    print("=" * 60)

    all_class_ids = sorted(set(model_names.keys()) | set(yaml_names.keys()) | set(gt_counts.keys()))
    rows = []
    for cid in all_class_ids:
        name = model_names.get(cid, yaml_names.get(cid, f"<unknown id {cid}>"))
        gt = gt_counts.get(cid, 0)
        m = per_class_metrics.get(cid)
        if m:
            status = "Evaluated"
            precision, recall, map50 = m["Precision"], m["Recall"], m["mAP@50"]
        elif gt == 0:
            status = "No GT instances in val split"
            precision = recall = map50 = float("nan")
        else:
            status = "Has GT but wasn't scored - check mismatch table above"
            precision = recall = map50 = float("nan")

        rows.append({
            "Class ID": cid,
            "Class Name": name,
            "GT Instances (val)": gt,
            "Precision": precision,
            "Recall": recall,
            "mAP@50": map50,
            "Status": status,
        })

    df = pd.DataFrame(rows).sort_values("Class ID")
    print("\n--- Full Class-Wise Report ---")
    print(df.to_string(index=False))

    output_csv = "symbol_evaluation_metrics_full.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nFull report written to '{output_csv}'")

    # Confusion matrix: the clearest way to SEE a class-id mismatch. A real
    # mismatch shows up as a consistent off-diagonal band rather than a clean
    # diagonal.
    try:
        cm = metrics.confusion_matrix
        n = cm.matrix.shape[0]
        labels = [model_names.get(i, str(i)) for i in range(n - 1)] + ["background"]
        cm_df = pd.DataFrame(cm.matrix, index=labels, columns=labels)
        cm_df.to_csv("confusion_matrix.csv")
        print("Confusion matrix written to 'confusion_matrix.csv' - open it "
              "and check whether errors cluster off the diagonal in a "
              "consistent pattern (strong sign of class-id misalignment).")
    except Exception as e:
        print(f"Could not export confusion matrix: {e}")

    return df


def evaluate_ocr_text(predicted_csv, ground_truth_csv, text_col="text", match_col="box_id"):
    """
    Optional: evaluate OCR/text-extraction accuracy separately from YOLO's
    detection metrics (YOLO's mAP does NOT tell you anything about whether
    the recognized *text* is correct, only whether a text region was found).

    Expects two CSVs joined on `match_col` (e.g. a box/line id you assign),
    each with a `text_col` column: one from your pipeline's OCR output, one
    manually verified ground truth. Requires: pip install jiwer
    """
    from jiwer import wer, cer

    pred = pd.read_csv(predicted_csv)
    gt = pd.read_csv(ground_truth_csv)
    merged = pred.merge(gt, on=match_col, suffixes=("_pred", "_gt"))

    merged["cer"] = merged.apply(
        lambda r: cer(str(r[f"{text_col}_gt"]), str(r[f"{text_col}_pred"])), axis=1
    )
    merged["wer"] = merged.apply(
        lambda r: wer(str(r[f"{text_col}_gt"]), str(r[f"{text_col}_pred"])), axis=1
    )

    print(f"Mean CER: {merged['cer'].mean():.4f}")
    print(f"Mean WER: {merged['wer'].mean():.4f}")
    merged.to_csv("ocr_evaluation_report.csv", index=False)
    return merged


if __name__ == "__main__":
    # The script absolutely needs your full model path here!
    MODEL_PATH = r"C:\Users\aryan\OneDrive\Desktop\Afzar Work\01 work\02 Working\Trained_model\pid-symbol-detector-final 3\pid-symbol-detector-final\models\best.pt"
    
    # And the full path to the data.yaml we just fixed
    DATA_YAML = r"C:\Users\aryan\OneDrive\Desktop\Afzar Work\01 work\02 Working\Trained_model\PID Full Dataset.v1-final-version-1.yolo26\data.yaml"

    evaluate_yolo_symbols(MODEL_PATH, DATA_YAML)
    # If/when you have OCR ground truth to check against, uncomment:
    # evaluate_ocr_text("pipeline_ocr_output.csv", "ocr_ground_truth.csv")