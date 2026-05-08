import cv2
import numpy as np

try:
    from .board import Cell
except ImportError:
    from board import Cell


class PieceColorClassifier:
    """
    Classifies holes by the sticker color on each piece:
      Robot  → teal/cyan sticker  (H 85-108, S≥90, V≥70)
      Human  → pink/magenta sticker (H 140-179, S≥80, V≥70)
      Empty  → neither sticker present
    """

    def __init__(
        self,
        teal_min_fraction: float = 0.04,
        teal_min_pixels: int = 30,
        pink_min_fraction: float = 0.06,
        pink_min_pixels: int = 40,
        debug: bool = False,
    ):
        self.teal_min_fraction = teal_min_fraction
        self.teal_min_pixels   = teal_min_pixels
        self.pink_min_fraction = pink_min_fraction
        self.pink_min_pixels   = pink_min_pixels
        self.debug = debug

    def classify(self, circle_bgr: np.ndarray) -> Cell:
        if circle_bgr is None or circle_bgr.size == 0:
            return Cell.EMPTY

        hsv = cv2.cvtColor(circle_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Ignore black pixels from the circular crop mask
        valid = np.any(circle_bgr > 15, axis=2)
        total = int(valid.sum())
        if total == 0:
            return Cell.EMPTY

        teal = valid & (h >= 85) & (h <= 108) & (s >= 90) & (v >= 70)
        pink = valid & (h >= 140) & (h <= 179) & (s >= 80) & (v >= 70)

        n_teal = int(teal.sum())
        n_pink = int(pink.sum())
        f_teal = n_teal / total
        f_pink = n_pink / total

        if self.debug:
            print(f"[Classifier] total={total} teal={n_teal}({f_teal:.3f}) pink={n_pink}({f_pink:.3f})")

        if n_teal >= self.teal_min_pixels and f_teal >= self.teal_min_fraction:
            if n_teal >= n_pink:
                return Cell.ROBOT

        if n_pink >= self.pink_min_pixels and f_pink >= self.pink_min_fraction:
            return Cell.HUMAN

        return Cell.EMPTY
