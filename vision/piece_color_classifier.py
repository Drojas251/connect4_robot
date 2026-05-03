import cv2
import numpy as np

try:
    from .board import Cell
except ImportError:
    from board import Cell


class PieceColorClassifier:
    def __init__(
        self,
        # --- Red (robot) ---
        red_min_fraction: float = 0.25,
        red_min_abs_pixels: int = 120,
        red_min_saturation: int = 130,
        red_min_value: int = 90,
        red_dominance_ratio: float = 2.0,

        # --- Yellow (human) ---
        yellow_min_fraction: float = 0.12,
        yellow_min_abs_pixels: int = 80,
        yellow_min_saturation: int = 80,
        yellow_min_value: int = 80,
        yellow_dominance_ratio: float = 1.3,

        debug: bool = False,
    ):
        self.red_min_fraction = red_min_fraction
        self.red_min_abs_pixels = red_min_abs_pixels
        self.red_min_saturation = red_min_saturation
        self.red_min_value = red_min_value
        self.red_dominance_ratio = red_dominance_ratio

        self.yellow_min_fraction = yellow_min_fraction
        self.yellow_min_abs_pixels = yellow_min_abs_pixels
        self.yellow_min_saturation = yellow_min_saturation
        self.yellow_min_value = yellow_min_value
        self.yellow_dominance_ratio = yellow_dominance_ratio

        self.debug = debug

    def classify(self, circle_bgr: np.ndarray) -> Cell:
        if circle_bgr is None or circle_bgr.size == 0:
            return Cell.EMPTY

        hsv = cv2.cvtColor(circle_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Ignore black pixels from circular mask
        crop_mask = np.any(circle_bgr > 10, axis=2)
        total_pixels = np.count_nonzero(crop_mask)

        if total_pixels == 0:
            return Cell.EMPTY

        # --- Valid masks ---
        red_valid = (
            crop_mask &
            (s >= self.red_min_saturation) &
            (v >= self.red_min_value)
        )

        yellow_valid = (
            crop_mask &
            (s >= self.yellow_min_saturation) &
            (v >= self.yellow_min_value)
        )

        # --- RED (tight, avoid false positives) ---
        red_mask_1 = (h >= 0) & (h <= 7)
        red_mask_2 = (h >= 173) & (h <= 179)
        red_mask = red_valid & (red_mask_1 | red_mask_2)

        # --- YELLOW (wider + more forgiving) ---
        # This is the KEY FIX for your issue
        yellow_mask = yellow_valid & (h >= 14) & (h <= 42)

        red_pixels = np.count_nonzero(red_mask)
        yellow_pixels = np.count_nonzero(yellow_mask)

        red_fraction = red_pixels / total_pixels
        yellow_fraction = yellow_pixels / total_pixels

        # --- Decision logic ---
        red_is_strong = (
            red_pixels >= self.red_min_abs_pixels and
            red_fraction >= self.red_min_fraction and
            red_pixels >= self.red_dominance_ratio * max(1, yellow_pixels)
        )

        yellow_is_strong = (
            yellow_pixels >= self.yellow_min_abs_pixels and
            yellow_fraction >= self.yellow_min_fraction and
            yellow_pixels >= self.yellow_dominance_ratio * max(1, red_pixels)
        )

        if self.debug:
            print(
                f"[Classifier] total={total_pixels} | "
                f"R: px={red_pixels}, frac={red_fraction:.2f} | "
                f"Y: px={yellow_pixels}, frac={yellow_fraction:.2f}"
            )

        if red_is_strong:
            return Cell.ROBOT

        if yellow_is_strong:
            return Cell.HUMAN

        return Cell.EMPTY