import time
import cv2
import numpy as np

from circle_grid_locator import CircleGridLocator
from piece_color_classifier import PieceColorClassifier
from board import Connect4Board, Cell
from roi_dataset_recorder import RoiDatasetRecorder
from learned_piece_classifier import LearnedPieceClassifier


CAMERA_INDEX = 1
REFRESH_SECONDS = 5
CELL_DISPLAY_SIZE = 80


def circular_crop_from_frame(frame, center_xy, radius):
    cx, cy = center_xy
    h, w = frame.shape[:2]

    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius)
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius)

    patch = frame[y_min:y_max, x_min:x_max].copy()
    if patch.size == 0:
        return None

    patch_h, patch_w = patch.shape[:2]
    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.circle(mask, (cx - x_min, cy - y_min), radius, 255, -1)
    return cv2.bitwise_and(patch, patch, mask=mask)


def apply_gravity(board: Connect4Board) -> Connect4Board:
    """Clear any cell that floats above an empty cell in the same column."""
    for col in range(len(board.grid[0])):
        seen_empty = False
        for row in range(len(board.grid)):   # row 0 = bottom
            if board.grid[row][col] == Cell.EMPTY:
                seen_empty = True
            elif seen_empty:
                board.grid[row][col] = Cell.EMPTY
    return board


def classify_board_from_fixed_holes(frame, holes, classifier, hole_crop_radius):
    board = Connect4Board.empty()
    hole_crops = []

    for hole in holes:
        crop = circular_crop_from_frame(frame, hole.frame_xy, hole_crop_radius)
        cell = classifier.classify(crop)
        board.grid[hole.row][hole.col] = cell

        label = "R" if cell == Cell.ROBOT else ("H" if cell == Cell.HUMAN else ".")
        hole_crops.append((hole.row, hole.col, crop, label))

    apply_gravity(board)

    # Recompute labels after gravity correction
    hole_crops = [
        (r, c, crop, "R" if board.grid[r][c] == Cell.ROBOT else
                      "H" if board.grid[r][c] == Cell.HUMAN else ".")
        for r, c, crop, _ in hole_crops
    ]

    return board, hole_crops


def make_debug_crop_grid(hole_crops, rows=6, cols=7):
    canvas = np.zeros(
        (rows * CELL_DISPLAY_SIZE, cols * CELL_DISPLAY_SIZE, 3),
        dtype=np.uint8,
    )

    for row, col, img, label in hole_crops:
        if img is None:
            continue

        resized = cv2.resize(img, (CELL_DISPLAY_SIZE, CELL_DISPLAY_SIZE))

        y0 = (rows - 1 - row) * CELL_DISPLAY_SIZE  # flip to visual top-down
        y1 = y0 + CELL_DISPLAY_SIZE
        x0 = col * CELL_DISPLAY_SIZE
        x1 = x0 + CELL_DISPLAY_SIZE

        canvas[y0:y1, x0:x1] = resized

        cv2.putText(
            canvas,
            f"{label} ({row},{col})",
            (x0 + 5, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    return canvas


def draw_fixed_hole_overlay(frame, holes, bbox_xyxy, hole_crop_radius, board=None):
    debug = frame.copy()

    x_min, y_min, x_max, y_max = bbox_xyxy
    cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    for hole in holes:
        label = f"{hole.row},{hole.col}"
        color = (255, 255, 0)

        if board is not None:
            cell = board.grid[hole.row][hole.col]
            if cell == Cell.ROBOT:
                label = f"R {hole.row},{hole.col}"
                color = (0, 0, 255)
            elif cell == Cell.HUMAN:
                label = f"H {hole.row},{hole.col}"
                color = (0, 255, 255)
            else:
                label = f". {hole.row},{hole.col}"
                color = (180, 180, 180)

        cv2.circle(debug, hole.frame_xy, hole_crop_radius, color, 2)
        cv2.circle(debug, hole.frame_xy, 3, color, -1)
        cv2.putText(
            debug,
            label,
            (hole.frame_xy[0] + 6, hole.frame_xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    return debug


def wait_with_window_updates_and_drain(cap, seconds):
    end_time = time.time() + seconds
    while time.time() < end_time:
        cap.grab()
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            return False
    return True


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    locator = CircleGridLocator()

    try:
        classifier = LearnedPieceClassifier(
            model_path="vision_piece_dataset/models/piece_random_forest.joblib",
            min_confidence=0.55,
            debug=True,
        )
    except Exception as e:
        print(f"LearnedPieceClassifier unavailable ({e}), falling back to PieceColorClassifier")
        classifier = PieceColorClassifier()

    recorder = RoiDatasetRecorder(
        dataset_dir="vision_piece_dataset",
        save_every_roi=True,
        save_on_state_change=True,
    )

    try:
        print("Searching for circle grid…")
        fixed_holes, fixed_bbox, grid_frame, mean_r = locator.find_grid(cap)
        hole_crop_radius = max(14, int(mean_r * 0.8))

        if not fixed_holes:
            print("Could not find a valid grid. Exiting.")
            return

        print(f"\nFixed grid locked ({len(fixed_holes)} holes, crop_r={hole_crop_radius}px).")
        print("BBox:", fixed_bbox)
        for hole in fixed_holes:
            print(f"R{hole.row} C{hole.col} | frame={hole.frame_xy}")

        grid_debug = draw_fixed_hole_overlay(grid_frame, fixed_holes, fixed_bbox, hole_crop_radius)
        cv2.imshow("Locked Fixed Grid", grid_debug)

        current_board = None
        print(f"\nStarting classification loop every {REFRESH_SECONDS} seconds.")
        print("Press ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                if not wait_with_window_updates_and_drain(cap, REFRESH_SECONDS):
                    break
                continue

            board, hole_crops = classify_board_from_fixed_holes(
                frame, fixed_holes, classifier, hole_crop_radius
            )
            recorder.save_rois(hole_crops, board, full_frame=frame)

            if not board.is_valid_physical_board():
                print("\nDetected board is not physically valid.")
                print(board.pretty())
            else:
                if current_board is not None:
                    move = current_board.infer_single_new_move(board)
                    if move is not None:
                        row, col, piece = move
                        print(f"\nNew move detected: row={row}, col={col}, piece={piece.value}")

                current_board = board
                print("\nDetected board:")
                print(current_board.pretty())

            overlay = draw_fixed_hole_overlay(
                frame, fixed_holes, fixed_bbox, hole_crop_radius, board
            )
            crop_grid = make_debug_crop_grid(hole_crops)

            cv2.imshow("Fixed Hole Classification Overlay", overlay)
            cv2.imshow("Classifier Circle Crops", crop_grid)

            if not wait_with_window_updates_and_drain(cap, REFRESH_SECONDS):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
