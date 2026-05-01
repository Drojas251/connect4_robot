from pathlib import Path
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


DATASET_DIR = Path("vision_piece_dataset")
MODEL_DIR = DATASET_DIR / "models"
MODEL_PATH = MODEL_DIR / "piece_random_forest.joblib"

LABELS = {
    "empty": 0,
    "yellow": 1,
    "red": 2,
}

ID_TO_LABEL = {
    0: "empty",
    1: "yellow",
    2: "red",
}


def extract_features(bgr):
    bgr = cv2.resize(bgr, (64, 64))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Ignore black masked-out pixels
    mask = np.any(bgr > 10, axis=2)

    if np.count_nonzero(mask) == 0:
        return np.zeros(96, dtype=np.float32)

    features = []

    for img in [bgr, hsv, lab]:
        for ch in range(3):
            vals = img[:, :, ch][mask]
            features.extend([
                float(np.mean(vals)),
                float(np.std(vals)),
                float(np.percentile(vals, 10)),
                float(np.percentile(vals, 50)),
                float(np.percentile(vals, 90)),
            ])

    # HSV hue histogram
    h = hsv[:, :, 0][mask]
    s = hsv[:, :, 1][mask]
    v = hsv[:, :, 2][mask]

    h_hist, _ = np.histogram(h, bins=18, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s, bins=8, range=(0, 256), density=True)
    v_hist, _ = np.histogram(v, bins=8, range=(0, 256), density=True)

    features.extend(h_hist.tolist())
    features.extend(s_hist.tolist())
    features.extend(v_hist.tolist())

    return np.array(features, dtype=np.float32)


def load_dataset():
    X = []
    y = []

    for label_name, label_id in LABELS.items():
        folder = DATASET_DIR / label_name

        for path in folder.glob("*.png"):
            img = cv2.imread(str(path))

            if img is None:
                continue

            X.append(extract_features(img))
            y.append(label_id)

    if not X:
        raise RuntimeError("No labeled images found.")

    return np.vstack(X), np.array(y)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    print(f"Loaded {len(y)} samples")
    for name, idx in LABELS.items():
        print(f"{name}: {np.sum(y == idx)}")

    if len(set(y)) < 3:
        raise RuntimeError("Need examples for empty, yellow, and red.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nReport:")
    print(classification_report(
        y_test,
        preds,
        target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)],
    ))

    payload = {
        "model": clf,
        "labels": ID_TO_LABEL,
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()