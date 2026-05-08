from pathlib import Path
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
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


_N_FEATURES = 81  # 45 stats + 34 histogram bins + 2 sticker fractions


# Brightness multipliers used for training augmentation (never applied to test set)
AUG_BRIGHTNESS = [0.30, 0.45, 0.60, 0.75, 1.25, 1.45, 1.60]


def _apply_brightness(bgr: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def extract_features(bgr):
    bgr = cv2.resize(bgr, (64, 64))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Ignore black masked-out pixels
    mask = np.any(bgr > 10, axis=2)

    if np.count_nonzero(mask) == 0:
        return np.zeros(_N_FEATURES, dtype=np.float32)

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

    # Explicit sticker-color fractions — the primary discriminating signal
    total = float(np.count_nonzero(mask))
    teal = mask & (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 108) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 70)
    pink = mask & (hsv[:, :, 0] >= 140) & (hsv[:, :, 0] <= 179) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 70)
    features.append(float(teal.sum()) / max(total, 1))
    features.append(float(pink.sum()) / max(total, 1))

    return np.array(features, dtype=np.float32)


def load_dataset(augment: bool = True, test_size: float = 0.25, seed: int = 42):
    """Return (X_train, y_train, X_test, y_test).

    Augmentation is applied only to the training split so test accuracy
    reflects real-world performance, not augmented copies of train images.
    """
    raw: list[tuple[np.ndarray, int]] = []

    for label_name, label_id in LABELS.items():
        folder = DATASET_DIR / label_name
        for path in sorted(folder.glob("*.png")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            raw.append((img, label_id))

    if not raw:
        raise RuntimeError("No labeled images found.")

    rng = np.random.RandomState(seed)
    idx = np.arange(len(raw))
    rng.shuffle(idx)
    split = int(len(idx) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]

    def _build(indices, with_aug):
        X, y = [], []
        for i in indices:
            img, label_id = raw[i]
            X.append(extract_features(img))
            y.append(label_id)
            if with_aug:
                for factor in AUG_BRIGHTNESS:
                    X.append(extract_features(_apply_brightness(img, factor)))
                    y.append(label_id)
        return np.vstack(X), np.array(y)

    X_train, y_train = _build(train_idx, augment)
    X_test,  y_test  = _build(test_idx,  False)   # never augment test set

    return X_train, y_train, X_test, y_test


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset(augment=True)

    print(f"Train samples (with augmentation): {len(y_train)}")
    print(f"Test  samples (originals only):    {len(y_test)}")
    for name, label_id in LABELS.items():
        print(f"  {name}: train={np.sum(y_train == label_id)}  test={np.sum(y_test == label_id)}")

    if len(set(y_train)) < 3:
        raise RuntimeError("Need examples for empty, yellow, and red.")

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