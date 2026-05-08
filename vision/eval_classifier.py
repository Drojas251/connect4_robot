"""
eval_classifier.py — measure the current model's accuracy on all labeled data.

Run from vision/:
    python eval_classifier.py
"""
from pathlib import Path
import cv2
import numpy as np
import joblib

try:
    from train_piece_classifier import extract_features, DATASET_DIR, MODEL_PATH, LABELS, ID_TO_LABEL
except ImportError:
    from vision.train_piece_classifier import extract_features, DATASET_DIR, MODEL_PATH, LABELS, ID_TO_LABEL


def main():
    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run train_piece_classifier.py first.")
        return

    payload = joblib.load(MODEL_PATH)
    model  = payload["model"]
    labels = payload["labels"]   # id → name

    correct = {name: 0 for name in LABELS}
    total   = {name: 0 for name in LABELS}
    errors  = []   # (true_label, pred_label, path)

    for label_name in LABELS:
        folder = DATASET_DIR / label_name
        for path in sorted(folder.glob("*.png")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            feat = extract_features(img).reshape(1, -1)
            pred_id   = int(model.predict(feat)[0])
            pred_name = labels[pred_id]

            total[label_name] += 1
            if pred_name == label_name:
                correct[label_name] += 1
            else:
                errors.append((label_name, pred_name, path.name))

    print("Per-class accuracy:")
    overall_correct = overall_total = 0
    for name in LABELS:
        n, c = total[name], correct[name]
        overall_correct += c
        overall_total   += n
        pct = 100 * c / n if n else 0
        print(f"  {name:8s}  {c}/{n}  ({pct:.1f}%)")

    print(f"\nOverall: {overall_correct}/{overall_total}  "
          f"({100*overall_correct/overall_total:.1f}%)" if overall_total else "")

    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for true, pred, name in errors[:20]:
            print(f"  true={true:8s}  pred={pred:8s}  {name}")
        if len(errors) > 20:
            print(f"  … and {len(errors)-20} more")
    else:
        print("\nNo misclassifications.")


if __name__ == "__main__":
    main()
