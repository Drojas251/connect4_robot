"""
CircleGridLocator — importable wrapper around the KF circle-grid detector.

Usage::

    locator = CircleGridLocator()
    holes, bbox_xyxy, frame, mean_r = locator.find_grid(cap)

`holes` is a list of Hole(row, col, frame_xy) using game convention
(row 0 = physical bottom of the board, row ROWS-1 = top).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from .detect_circle_grid import (
        detect_blue_board,
        detect_circles,
        fit_grid,
        build_measurements,
        all_expected_centers,
        GridKalman,
        KF_SETTLE_STD,
        EXPECTED_ROWS,
        EXPECTED_COLS,
    )
except ImportError:
    from detect_circle_grid import (
        detect_blue_board,
        detect_circles,
        fit_grid,
        build_measurements,
        all_expected_centers,
        GridKalman,
        KF_SETTLE_STD,
        EXPECTED_ROWS,
        EXPECTED_COLS,
    )


@dataclass
class Hole:
    row: int               # game convention: 0 = bottom row
    col: int
    frame_xy: tuple[int, int]


class CircleGridLocator:
    """
    Detects and KF-tracks the Connect4 hole grid from a camera frame stream.

    Row convention: the detector internally numbers rows top-down (ri=0 is the
    topmost visual row).  Holes returned by find_grid() are remapped so that
    row 0 is the physical bottom of the board (game/gravity convention).
    """

    def __init__(self):
        self._kf = GridKalman()
        self.last_bbox = None
        self.last_gp: Optional[dict] = None

    def process_frame(self, frame: np.ndarray):
        """
        Run one frame through the detector pipeline.
        Returns (kf_output, bbox_xywh, grid_params).
        """
        bbox, _ = detect_blue_board(frame)
        circles = detect_circles(frame, bbox)
        assignments, gp = fit_grid(circles)
        meas = build_measurements(circles, assignments, gp)
        expected = all_expected_centers(gp) if gp else None
        kf_output = self._kf.step(meas, expected_centers=expected)
        self.last_bbox = bbox
        self.last_gp = gp
        return kf_output, bbox, gp

    def all_settled(self, kf_output: dict) -> bool:
        """True when all expected slots have been seen and their KF converged."""
        if len(kf_output) < EXPECTED_ROWS * EXPECTED_COLS:
            return False
        return all(std < KF_SETTLE_STD for _, std in kf_output.values())

    def find_grid(
        self,
        cap: cv2.VideoCapture,
        max_frames: int = 300,
        running_fn=None,
    ) -> tuple[list[Hole], tuple, Optional[np.ndarray], float]:
        """
        Read frames from `cap` until the KF grid fully settles.

        running_fn: optional callable → bool; return False to abort early.

        Returns (holes, bbox_xyxy, last_frame, mean_r).
        holes is empty if the grid was never found.
        """
        last_frame = None
        for i in range(max_frames):
            if running_fn is not None and not running_fn():
                break
            ret, frame = cap.read()
            if not ret:
                continue
            last_frame = frame

            kf_output, bbox, gp = self.process_frame(frame)

            if i % 30 == 0:
                print(
                    f"[CircleGridLocator] frame {i}: "
                    f"{len(kf_output)}/{EXPECTED_ROWS * EXPECTED_COLS} slots, "
                    f"settled={self.all_settled(kf_output)}"
                )

            if bbox is not None and self.all_settled(kf_output):
                print(f"[CircleGridLocator] Grid settled after {i} frames")
                holes = self._build_holes(kf_output)
                bx, by, bw, bh = bbox
                mean_r = float(gp["mean_r"]) if gp else 30.0
                return holes, (bx, by, bx + bw, by + bh), last_frame, mean_r

        print(f"[CircleGridLocator] Grid not settled after {max_frames} frames")
        return [], (0, 0, 0, 0), last_frame, 30.0

    def _build_holes(self, kf_output: dict) -> list[Hole]:
        holes = []
        for (ri, ci), (pos, _) in kf_output.items():
            # Remap: ri=0 is topmost visual row → game row ROWS-1 (top)
            game_row = EXPECTED_ROWS - 1 - ri
            holes.append(Hole(row=game_row, col=ci, frame_xy=pos))
        holes.sort(key=lambda h: (h.row, h.col))
        return holes
