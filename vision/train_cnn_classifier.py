"""
train_cnn_classifier.py — train the small CNN piece classifier.

Run from vision/:
    python train_cnn_classifier.py
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report

try:
    from cnn_piece_classifier import build_model, IMG_SIZE
except ImportError:
    from vision.cnn_piece_classifier import build_model, IMG_SIZE


DATASET_DIR = Path("vision_piece_dataset")
MODEL_DIR   = DATASET_DIR / "models"
MODEL_PATH  = MODEL_DIR / "piece_cnn.pt"

LABELS     = {"empty": 0, "yellow": 1, "red": 2}
ID_TO_LABEL = {0: "empty", 1: "yellow", 2: "red"}

EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42


# ── augmentation ─────────────────────────────────────────────────────────────

# Keep hue jitter tiny — the sticker hue IS the classification signal.
_train_tf = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.3, hue=0.03),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_val_tf = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class _PieceDataset(Dataset):
    def __init__(self, images: list, labels: list, transform):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        bgr = self.images[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb), self.labels[idx]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_raw():
    images, labels = [], []
    for label_name, label_id in LABELS.items():
        folder = DATASET_DIR / label_name
        for path in sorted(folder.glob("*.png")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_id)
    if not images:
        raise RuntimeError("No labeled images found in vision_piece_dataset/.")
    return images, labels


def _split(images, labels, test_frac=0.25, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    split = int(len(idx) * (1 - test_frac))
    train_i, test_i = idx[:split], idx[split:]
    return (
        [images[i] for i in train_i], [labels[i] for i in train_i],
        [images[i] for i in test_i],  [labels[i] for i in test_i],
    )


# ── training ──────────────────────────────────────────────────────────────────

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    images, labels = _load_raw()
    train_imgs, train_lbls, test_imgs, test_lbls = _split(images, labels)

    print(f"Train: {len(train_imgs)}  Test: {len(test_imgs)}")
    for name, lid in LABELS.items():
        tr = sum(l == lid for l in train_lbls)
        te = sum(l == lid for l in test_lbls)
        print(f"  {name}: train={tr}  test={te}")

    train_ds = _PieceDataset(train_imgs, train_lbls, _train_tf)
    test_ds  = _PieceDataset(test_imgs,  test_lbls,  _val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = build_model(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for X, y in test_dl:
                    preds = model(X.to(device)).argmax(1)
                    correct += (preds == y.to(device)).sum().item()
                    total   += y.size(0)
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={running_loss/len(train_dl):.4f}"
                  f"  val_acc={correct/total:.4f}")

    # Final evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in test_dl:
            preds = model(X.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())

    print("\nConfusion matrix:")
    print(confusion_matrix(all_true, all_preds))
    print("\nReport:")
    print(classification_report(
        all_true, all_preds,
        target_names=[ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)],
    ))

    torch.save({
        "model_state": model.state_dict(),
        "labels":      ID_TO_LABEL,
        "img_size":    IMG_SIZE,
    }, MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
