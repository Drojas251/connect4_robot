from pathlib import Path
from datetime import datetime
import cv2


class RoiDatasetRecorder:
    def __init__(
        self,
        dataset_dir="vision_piece_dataset",
        save_every_roi=True,
        save_on_state_change=True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.unlabeled_dir = self.dataset_dir / "unlabeled"
        self.save_every_roi = save_every_roi
        self.save_on_state_change = save_on_state_change
        self.last_board_signature = None

        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)

    def board_signature(self, board):
        return tuple(tuple(cell.value for cell in row) for row in board.grid)

    def should_save(self, board):
        if self.save_every_roi:
            return True

        if not self.save_on_state_change:
            return False

        sig = self.board_signature(board)

        if sig != self.last_board_signature:
            self.last_board_signature = sig
            return True

        return False

    def save_rois(self, hole_crops, board=None):
        if board is not None and not self.should_save(board):
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        for row, col, crop, predicted_label in hole_crops:
            if crop is None or crop.size == 0:
                continue

            filename = f"{ts}_r{row}_c{col}_pred_{predicted_label}.png"
            out_path = self.unlabeled_dir / filename
            cv2.imwrite(str(out_path), crop)

        print(f"Saved ROI crops to {self.unlabeled_dir}")