from pathlib import Path
import shutil
import cv2
import numpy as np


DATASET_DIR = Path("vision_piece_dataset")
UNLABELED_DIR = DATASET_DIR / "unlabeled"
FRAMES_DIR = DATASET_DIR / "frames"

LABEL_DIRS = {
    "r": DATASET_DIR / "red",
    "y": DATASET_DIR / "yellow",
    "e": DATASET_DIR / "empty",
}

CELL = 80   # px per cell in the board context window
ROWS = 6
COLS = 7


def parse_filename(name: str):
    """Return (timestamp_prefix, row, col) from a crop filename."""
    stem = Path(name).stem  # e.g. 20260430_060048_798636_r0_c3_pred_.
    parts = stem.split("_")
    ts = "_".join(parts[:3])
    row = col = None
    for p in parts:
        if p.startswith("r") and p[1:].isdigit():
            row = int(p[1:])
        if p.startswith("c") and p[1:].isdigit():
            col = int(p[1:])
    return ts, row, col


def build_board_context(current_path: Path, all_images: list[Path]) -> np.ndarray:
    """
    Build a ROWS×COLS grid of crops from the same timestamp as current_path.
    The current crop is highlighted with a cyan border.
    Falls back to a full-frame JPEG if one was saved alongside the crops.
    """
    ts, cur_row, cur_col = parse_filename(current_path.name)

    # Try full frame first
    frame_path = FRAMES_DIR / f"{ts}.jpg"
    if frame_path.exists():
        full = cv2.imread(str(frame_path))
        if full is not None:
            # Scale to fit in a reasonable window (max 960 wide)
            h, w = full.shape[:2]
            scale = min(1.0, 960 / w)
            disp = cv2.resize(full, (int(w * scale), int(h * scale)))
            return disp

    # Fallback: composite grid from same-timestamp crops
    siblings = {p for p in all_images if parse_filename(p.name)[0] == ts}
    canvas = np.zeros((ROWS * CELL, COLS * CELL, 3), dtype=np.uint8)

    for p in siblings:
        _, r, c = parse_filename(p.name)
        if r is None or c is None:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        cell_img = cv2.resize(img, (CELL, CELL))
        y0, x0 = (ROWS - 1 - r) * CELL, c * CELL  # flip rows: row0=bottom→bottom of canvas
        canvas[y0:y0 + CELL, x0:x0 + CELL] = cell_img

        # Grid line
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL - 1, y0 + CELL - 1), (60, 60, 60), 1)

        # Label
        _, pr, pc = parse_filename(p.name)
        cv2.putText(canvas, f"r{pr}c{pc}", (x0 + 3, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Highlight current crop
    if cur_row is not None and cur_col is not None:
        y0 = (ROWS - 1 - cur_row) * CELL
        x0 = cur_col * CELL
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL - 1, y0 + CELL - 1), (255, 220, 0), 3)

    return canvas


def main():
    for d in LABEL_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(UNLABELED_DIR.glob("*.png"))
        + list(UNLABELED_DIR.glob("*.jpg"))
        + list(UNLABELED_DIR.glob("*.jpeg"))
    )

    if not images:
        print("No unlabeled images found.")
        return

    print(f"{len(images)} unlabeled images.")
    print("Controls:  r=red/robot   y=yellow/human   e=empty   s=skip   q=quit")

    idx = 0

    while idx < len(images):
        path = images[idx]
        img = cv2.imread(str(path))

        if img is None:
            idx += 1
            continue

        # ── ROI crop window ──────────────────────────────────────────────
        display = cv2.resize(img, (220, 220), interpolation=cv2.INTER_NEAREST)
        cv2.putText(display, f"{idx + 1}/{len(images)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, path.name[:28], (10, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("Label ROI: r=red  y=yellow  e=empty  s=skip  q=quit", display)

        # ── Board context window ─────────────────────────────────────────
        context = build_board_context(path, images)
        cv2.imshow("Board context", context)

        key = cv2.waitKey(0) & 0xFF
        char = chr(key).lower() if key < 128 else ""

        if char == "q":
            break

        if char == "s":
            idx += 1
            continue

        if char in LABEL_DIRS:
            dst = LABEL_DIRS[char] / path.name
            shutil.move(str(path), str(dst))
            print(f"{path.name} -> {dst.parent.name}")
            images.pop(idx)  # remove from list; don't advance idx
            continue

        print("Unknown key. Use r, y, e, s, q.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
