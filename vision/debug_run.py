"""
Standalone vision debug runner.

Opens the camera, locks onto the AprilTag grid, then continuously classifies
the board and prints state to the terminal.  No HTTP calls are made — this
is purely for verifying the vision pipeline works before plugging it into the
full system.

Usage (from repo root /home/aft/):
    /home/aft/dsr-motion/api/python/venv/bin/python3 -m connect4_robot.vision.debug_run

Optional env vars:
    CAMERA=0        camera index (default 1)
    CLASSIFIER=hsv  use HSV color classifier instead of learned model (default: auto)
    VERBOSE=1       print per-cell scores every frame
"""
from __future__ import annotations

import os
import sys
import time
import signal

import cv2
import numpy as np

try:
    from .connect4_tag_grid import Connect4TagGridDetector
    from .piece_color_classifier import PieceColorClassifier
    from .learned_piece_classifier import LearnedPieceClassifier
    from connect4_robot.game_engine.board import Connect4Board, Cell
except ImportError:
    from connect4_tag_grid import Connect4TagGridDetector
    from piece_color_classifier import PieceColorClassifier
    from learned_piece_classifier import LearnedPieceClassifier
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from game_engine.board import Connect4Board, Cell


CAMERA_INDEX = int(os.environ.get("CAMERA", 1))
HOLE_CROP_RADIUS = 18
MAX_GRID_SEARCH_FRAMES = 300
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

# ANSI colours
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def ts() -> str:
    return time.strftime("%H:%M:%S")


def board_lines(board: Connect4Board) -> list[str]:
    """Return the board as coloured unicode lines."""
    lines = []
    for r in reversed(range(board.grid.__len__())):
        row_cells = []
        for cell in board.grid[r]:
            if cell == Cell.HUMAN:
                row_cells.append(f"{YELLOW}H{RESET}")
            elif cell == Cell.ROBOT:
                row_cells.append(f"{RED}R{RESET}")
            else:
                row_cells.append(f"{DIM}.{RESET}")
        lines.append("  " + " ".join(row_cells))
    lines.append("  " + " ".join(str(c) for c in range(len(board.grid[0]))))
    return lines


def print_board(board: Connect4Board, label: str = ""):
    if label:
        print(f"{CYAN}{label}{RESET}")
    for line in board_lines(board):
        print(line)


def board_eq(a: Connect4Board, b: Connect4Board) -> bool:
    return a.to_strings_top_down() == b.to_strings_top_down()


def circular_crop(frame, center_xy, radius):
    cx, cy = center_xy
    h, w = frame.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius)
    y0, y1 = max(0, cy - radius), min(h, cy + radius)
    patch = frame[y0:y1, x0:x1].copy()
    if patch.size == 0:
        return None
    ph, pw = patch.shape[:2]
    mask = np.zeros((ph, pw), np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), radius, 255, -1)
    return cv2.bitwise_and(patch, patch, mask=mask)


def classify_board(frame, holes, classifier) -> Connect4Board:
    board = Connect4Board.empty()
    for hole in holes:
        crop = circular_crop(frame, hole.frame_xy, HOLE_CROP_RADIUS)
        cell = classifier.classify(crop) if crop is not None else Cell.EMPTY
        board.grid[hole.row][hole.col] = cell
    return board


def find_grid(cap: cv2.VideoCapture, detector: Connect4TagGridDetector):
    print(f"[{ts()}] Searching for AprilTag grid (up to {MAX_GRID_SEARCH_FRAMES} frames)…")
    for i in range(MAX_GRID_SEARCH_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        result = detector.detect_from_frame(frame)
        if result is not None and len(result.holes) > 0:
            print(f"[{ts()}] {BOLD}Grid found{RESET} on frame {i} — {len(result.holes)} holes")
            return result
        if i % 30 == 0 and i > 0:
            print(f"[{ts()}]   … still searching (frame {i})")
    return None


def make_classifier():
    force_hsv = os.environ.get("CLASSIFIER", "").lower() == "hsv"
    if force_hsv:
        print(f"[{ts()}] Using {CYAN}PieceColorClassifier (HSV){RESET}")
        return PieceColorClassifier(debug=VERBOSE)
    try:
        clf = LearnedPieceClassifier(debug=VERBOSE)
        print(f"[{ts()}] Using {CYAN}LearnedPieceClassifier{RESET}")
        return clf
    except Exception as e:
        print(f"[{ts()}] LearnedPieceClassifier unavailable ({e}), falling back to HSV")
        return PieceColorClassifier(debug=VERBOSE)


def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Connect4 Vision Debug Runner{RESET}")
    print(f"  camera={CAMERA_INDEX}  verbose={VERBOSE}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {CAMERA_INDEX}", file=sys.stderr)
        sys.exit(1)

    classifier = make_classifier()
    detector = Connect4TagGridDetector()

    grid = find_grid(cap, detector)
    if grid is None:
        print("ERROR: grid not found after exhausting search frames", file=sys.stderr)
        cap.release()
        sys.exit(1)

    holes = grid.holes
    print(f"\n{BOLD}Grid locked.{RESET}  Entering classification loop — Ctrl+C to stop.\n")

    last_board = Connect4Board.empty()
    last_print_time = 0.0
    frame_count = 0
    fps_t0 = time.time()

    def _sigint(sig, frame):
        print(f"\n[{ts()}] Stopping.")
        cap.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{ts()}] WARNING: failed to read frame")
            time.sleep(0.1)
            continue

        frame_count += 1
        board = classify_board(frame, holes, classifier)
        now = time.time()

        board_changed = not board_eq(board, last_board)

        # Print on change, or at least every 5 s
        if board_changed or (now - last_print_time) >= 5.0:
            rows = board.to_strings_top_down()
            fps = frame_count / max(0.001, now - fps_t0)

            if board_changed:
                print(f"[{ts()}] {BOLD}BOARD CHANGED{RESET}  frame={frame_count}  fps={fps:.1f}")
                # Show diff: which cells changed
                for r_idx, (old_row, new_row) in enumerate(
                    zip(last_board.to_strings_top_down(), rows)
                ):
                    for c_idx, (old_v, new_v) in enumerate(zip(old_row, new_row)):
                        if old_v != new_v:
                            colour = RED if new_v == "R" else (YELLOW if new_v == "H" else DIM)  # R=red, H=yellow
                            print(f"  cell row={r_idx} col={c_idx}: {old_v} → {colour}{new_v}{RESET}")
            else:
                print(f"[{ts()}] frame={frame_count}  fps={fps:.1f}  (no change)")

            print_board(board)

            # Print what would be sent over the wire
            print(f"  {DIM}would publish: board={rows}{RESET}")

            if not board.is_valid_physical_board():
                print(f"  {RED}WARNING: board is not physically valid (floating pieces){RESET}")

            winner = board.check_winner()
            if winner:
                print(f"  {BOLD}WINNER: {winner.value}{RESET}")

            print()
            last_board = board
            last_print_time = now

        time.sleep(0.05)  # ~20 fps poll


if __name__ == "__main__":
    main()
