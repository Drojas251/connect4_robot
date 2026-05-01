import cv2
import numpy as np

from board import Connect4Board, Cell
from config import BOARD

from connect4_tag_grid import Connect4TagGridDetector
from piece_color_classifier import PieceColorClassifier


class VisionBoardDetector:
    def __init__(
        self,
        hole_crop_radius: int = 18,
        grid_detector: Connect4TagGridDetector | None = None,
        classifier: PieceColorClassifier | None = None,
    ):
        self.hole_crop_radius = hole_crop_radius
        self.grid_detector = grid_detector or Connect4TagGridDetector()
        self.classifier = classifier or PieceColorClassifier()

    def detect_board_from_frame(self, frame: np.ndarray) -> tuple[Connect4Board, object] | None:
        """
        Returns:
            (Connect4Board, grid_result)

        Assumes grid_result.holes uses:
            row 0 = bottom row
            row 4 = top row
        """

        grid_result = self.grid_detector.detect_from_frame(frame)

        if grid_result is None:
            return None

        board = Connect4Board.empty()
        roi = grid_result.roi_image

        for hole in grid_result.holes:
            if hole.row >= BOARD.rows or hole.col >= BOARD.cols:
                continue

            circle_crop = self._circular_crop_from_roi(
                roi,
                hole.roi_xy,
                self.hole_crop_radius,
            )

            cell = self.classifier.classify(circle_crop)

            # Your convention:
            # row 0 = bottom row
            # row 4 = top row
            board.grid[hole.row][hole.col] = cell

        return board, grid_result

    def _circular_crop_from_roi(self, roi, center_xy, radius):
        cx, cy = center_xy

        h, w = roi.shape[:2]

        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius)

        patch = roi[y_min:y_max, x_min:x_max].copy()

        if patch.size == 0:
            return None

        patch_h, patch_w = patch.shape[:2]

        local_cx = cx - x_min
        local_cy = cy - y_min

        mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)

        circular_patch = cv2.bitwise_and(patch, patch, mask=mask)

        return circular_patch

    def draw_debug(self, grid_result, board: Connect4Board):
        roi_debug = grid_result.roi_image.copy()

        for hole in grid_result.holes:
            if hole.row >= BOARD.rows or hole.col >= BOARD.cols:
                continue

            cell = board.grid[hole.row][hole.col]

            if cell == Cell.ROBOT:
                color = (0, 0, 255)
                label = "R"
            elif cell == Cell.HUMAN:
                color = (0, 255, 255)
                label = "H"
            else:
                color = (180, 180, 180)
                label = "."

            cv2.circle(
                roi_debug,
                hole.roi_xy,
                self.hole_crop_radius,
                color,
                2,
            )

            cv2.putText(
                roi_debug,
                f"{label} {hole.row},{hole.col}",
                (hole.roi_xy[0] + 6, hole.roi_xy[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        return roi_debug