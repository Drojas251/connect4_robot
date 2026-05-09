"""
cnn_piece_classifier.py — CNN-based piece classifier.

Shared between training (model definition) and inference (CnnPieceClassifier).
"""
from pathlib import Path

import cv2
import numpy as np

try:
    from .board import Cell
except ImportError:
    from board import Cell

IMG_SIZE = 64

# Normalisation constants (channel mean/std in [0,1] RGB space)
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _bgr_to_tensor(bgr: np.ndarray) -> "torch.Tensor":
    """Resize, convert BGR→RGB, normalise, return (1,3,H,W) float32 tensor."""
    import torch
    img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD                          # HWC
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # 1CHW


def build_model(n_classes: int = 3):
    """Return an untrained PieceCNN. Import torch before calling."""
    import torch.nn as nn

    return nn.Sequential(
        # block 1 — 64→32
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2),
        # block 2 — 32→16
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        # block 3 — 16→8
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        # block 4 — 8→2×2 global pool
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        # head
        nn.Flatten(),
        nn.Linear(128 * 2 * 2, 128), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(128, n_classes),
    )


_DEFAULT_MODEL = Path(__file__).parent / "vision_piece_dataset" / "models" / "piece_cnn.pt"


class CnnPieceClassifier:
    """Same classify(circle_bgr) → Cell interface as LearnedPieceClassifier."""

    def __init__(
        self,
        model_path=None,
        debug: bool = False,
    ):
        import torch
        self.debug = debug
        path = Path(model_path) if model_path is not None else _DEFAULT_MODEL

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.labels: dict = payload["labels"]
        self.model = build_model(n_classes=len(self.labels))
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self._torch = torch

    def classify(self, circle_bgr: np.ndarray) -> Cell:
        if circle_bgr is None or circle_bgr.size == 0:
            return Cell.EMPTY

        tensor = _bgr_to_tensor(circle_bgr)
        with self._torch.no_grad():
            logits = self.model(tensor)[0]
            probs  = self._torch.softmax(logits, dim=0).numpy()

        pred_id    = int(probs.argmax())
        confidence = float(probs[pred_id])
        label      = self.labels[pred_id]

        if self.debug:
            prob_str = {self.labels[int(i)]: float(p) for i, p in enumerate(probs)}
            print(f"[CNN] label={label}, conf={confidence:.2f}, probs={prob_str}")

        if label == "red":
            return Cell.ROBOT
        if label == "yellow":
            return Cell.HUMAN
        return Cell.EMPTY
