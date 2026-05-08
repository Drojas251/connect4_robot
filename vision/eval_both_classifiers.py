"""
eval_both_classifiers.py — compare RF and CNN on all labeled data.

Run from vision/:
    python eval_both_classifiers.py
"""
from pathlib import Path
import cv2
import numpy as np
import joblib

try:
    from train_piece_classifier import extract_features, DATASET_DIR, LABELS, ID_TO_LABEL
    from cnn_piece_classifier import CnnPieceClassifier, _bgr_to_tensor
except ImportError:
    from vision.train_piece_classifier import extract_features, DATASET_DIR, LABELS, ID_TO_LABEL
    from vision.cnn_piece_classifier import CnnPieceClassifier, _bgr_to_tensor

RF_MODEL_PATH  = DATASET_DIR / "models" / "piece_random_forest.joblib"
CNN_MODEL_PATH = DATASET_DIR / "models" / "piece_cnn.pt"

MIN_RF_CONFIDENCE = 0.35


def _eval_rf(paths_labels):
    payload = joblib.load(RF_MODEL_PATH)
    model   = payload["model"]
    labels  = payload["labels"]

    correct = {n: 0 for n in LABELS}
    total   = {n: 0 for n in LABELS}
    errors  = []

    for path, true_name in paths_labels:
        img = cv2.imread(str(path))
        if img is None:
            continue
        feat      = extract_features(img).reshape(1, -1)
        probs     = model.predict_proba(feat)[0]
        pred_id   = int(probs.argmax())
        confidence = float(probs[pred_id])
        pred_name  = labels[pred_id]

        if confidence < MIN_RF_CONFIDENCE:
            pred_name = "empty"

        total[true_name] += 1
        if pred_name == true_name:
            correct[true_name] += 1
        else:
            errors.append((true_name, pred_name, path.name))

    return correct, total, errors


def _eval_cnn(paths_labels):
    import torch

    try:
        cnn = CnnPieceClassifier(model_path=str(CNN_MODEL_PATH), debug=False)
    except Exception as e:
        print(f"  CNN model unavailable: {e}")
        return None, None, None

    label_map = {"empty": "empty", "yellow": "yellow", "red": "red"}

    correct = {n: 0 for n in LABELS}
    total   = {n: 0 for n in LABELS}
    errors  = []

    for path, true_name in paths_labels:
        img = cv2.imread(str(path))
        if img is None:
            continue
        tensor = _bgr_to_tensor(img)
        with torch.no_grad():
            probs    = torch.softmax(cnn.model(tensor)[0], dim=0).numpy()
        pred_id   = int(probs.argmax())
        pred_name = cnn.labels[pred_id]

        total[true_name] += 1
        if pred_name == true_name:
            correct[true_name] += 1
        else:
            errors.append((true_name, pred_name, path.name))

    return correct, total, errors


def _print_results(name, correct, total, errors):
    print(f"\n── {name} ──")
    overall_c = overall_t = 0
    for label_name in LABELS:
        n, c = total[label_name], correct[label_name]
        overall_c += c
        overall_t += n
        pct = 100 * c / n if n else 0
        print(f"  {label_name:8s}  {c}/{n}  ({pct:.1f}%)")
    if overall_t:
        print(f"  {'OVERALL':8s}  {overall_c}/{overall_t}  "
              f"({100*overall_c/overall_t:.1f}%)")
    if errors:
        print(f"\n  Misclassified ({len(errors)}):")
        for true, pred, name_ in errors[:20]:
            print(f"    true={true:8s}  pred={pred:8s}  {name_}")
        if len(errors) > 20:
            print(f"    … and {len(errors)-20} more")
    else:
        print("  No misclassifications.")


def main():
    # Collect (path, label_name) for all labeled images
    paths_labels = []
    for label_name in LABELS:
        folder = DATASET_DIR / label_name
        for path in sorted(folder.glob("*.png")):
            paths_labels.append((path, label_name))

    print(f"Total labeled images: {len(paths_labels)}")

    if RF_MODEL_PATH.exists():
        rf_correct, rf_total, rf_errors = _eval_rf(paths_labels)
        _print_results("Random Forest", rf_correct, rf_total, rf_errors)
    else:
        print("RF model not found — skipping.")

    if CNN_MODEL_PATH.exists():
        cnn_correct, cnn_total, cnn_errors = _eval_cnn(paths_labels)
        if cnn_correct is not None:
            _print_results("CNN", cnn_correct, cnn_total, cnn_errors)
    else:
        print("\nCNN model not found — train it first with train_cnn_classifier.py.")


if __name__ == "__main__":
    main()
