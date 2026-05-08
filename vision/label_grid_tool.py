"""
label_grid_tool.py — label a full board frame at a time.

Shows a 6×7 grid of hole crops with predicted labels overlaid.
Click a cell to cycle its label (empty → robot → human → empty).
Controls:
  click   cycle label on that cell
  Enter   accept this frame → move all crops to their label dirs
  s       skip this frame (leave in unlabeled)
  q       quit
"""
from pathlib import Path
import shutil
import cv2
import numpy as np

DATASET_DIR  = Path("vision_piece_dataset")
UNLABELED_DIR = DATASET_DIR / "unlabeled"
FRAMES_DIR   = DATASET_DIR / "frames"
LABEL_DIRS   = {
    ".": DATASET_DIR / "empty",
    "R": DATASET_DIR / "red",
    "H": DATASET_DIR / "yellow",
}

ROWS, COLS = 6, 7
CELL = 100          # px per cell in the grid display
LABEL_CYCLE = [".", "R", "H"]

# Visual colours for each label (BGR)
LABEL_COLOR = {".": (120, 120, 120), "R": (60, 60, 220), "H": (0, 200, 255)}
LABEL_TEXT  = {".": "empty", "R": "robot", "H": "human"}


# ── filename helpers ───────────────────────────────────────────────────────

def parse_filename(name: str):
    """Return (ts, row, col, pred_label) from a crop filename."""
    stem  = Path(name).stem   # e.g. 20260430_060048_798636_r0_c3_pred_R
    parts = stem.split("_")
    ts = "_".join(parts[:3])
    row = col = None
    pred = "."
    for i, p in enumerate(parts):
        if p.startswith("r") and p[1:].isdigit():
            row = int(p[1:])
        if p.startswith("c") and p[1:].isdigit():
            col = int(p[1:])
        if p == "pred" and i + 1 < len(parts):
            raw = parts[i + 1]
            if raw in ("R", "H"):
                pred = raw
    return ts, row, col, pred


def group_by_ts(images: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in images:
        ts, row, col, _ = parse_filename(p.name)
        if row is None or col is None:
            continue
        groups.setdefault(ts, []).append(p)
    return dict(sorted(groups.items()))


# ── display ────────────────────────────────────────────────────────────────

def build_grid_image(crops: dict[tuple, np.ndarray], labels: dict[tuple, str],
                     highlight: tuple | None = None) -> np.ndarray:
    """Build a ROWS×COLS grid canvas from crop images and current labels."""
    canvas = np.zeros((ROWS * CELL, COLS * CELL, 3), dtype=np.uint8)

    for ri in range(ROWS):
        for ci in range(COLS):
            # visual row 0 = top of canvas = game row ROWS-1
            game_row = ROWS - 1 - ri
            x0, y0 = ci * CELL, ri * CELL
            x1, y1 = x0 + CELL, y0 + CELL

            crop = crops.get((game_row, ci))
            label = labels.get((game_row, ci), ".")
            color = LABEL_COLOR[label]

            if crop is not None:
                cell_img = cv2.resize(crop, (CELL, CELL))
                canvas[y0:y1, x0:x1] = cell_img
            else:
                canvas[y0:y1, x0:x1] = 40  # dark grey placeholder

            # Colored border
            thickness = 4 if (game_row, ci) == highlight else 2
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, thickness)

            # Label text
            cv2.putText(canvas, LABEL_TEXT[label],
                        (x0 + 4, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(canvas, f"r{game_row}c{ci}",
                        (x0 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    return canvas


def cell_from_click(x: int, y: int) -> tuple[int, int] | None:
    """Convert canvas click coordinates to (game_row, col)."""
    ci = x // CELL
    ri = y // CELL
    if 0 <= ci < COLS and 0 <= ri < ROWS:
        return (ROWS - 1 - ri, ci)
    return None


# ── main ──────────────────────────────────────────────────────────────────

clicked_cell = None

def on_mouse(event, x, y, flags, param):
    global clicked_cell
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_cell = cell_from_click(x, y)


def main():
    for d in LABEL_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(UNLABELED_DIR.glob("*.png"))
    if not images:
        print("No unlabeled images found.")
        return

    groups = group_by_ts(images)
    print(f"{len(images)} images in {len(groups)} frame groups.")
    print("Controls: click=cycle label  Enter=accept  s=skip  q=quit")

    cv2.namedWindow("Grid labels")
    cv2.setMouseCallback("Grid labels", on_mouse)

    global clicked_cell

    group_list = list(groups.items())
    g_idx = 0

    while g_idx < len(group_list):
        ts, paths = group_list[g_idx]

        # Load crops and initialise labels from filename predictions
        crops: dict[tuple, np.ndarray] = {}
        labels: dict[tuple, str] = {}
        path_map: dict[tuple, Path] = {}

        for p in paths:
            _, row, col, pred = parse_filename(p.name)
            if row is None or col is None:
                continue
            img = cv2.imread(str(p))
            crops[(row, col)] = img
            labels[(row, col)] = pred
            path_map[(row, col)] = p

        if not crops:
            g_idx += 1
            continue

        # Optional full-frame context window
        frame_path = FRAMES_DIR / f"{ts}.jpg"
        if frame_path.exists():
            full = cv2.imread(str(frame_path))
            if full is not None:
                h, w = full.shape[:2]
                scale = min(1.0, 800 / w)
                cv2.imshow("Full frame", cv2.resize(full, (int(w*scale), int(h*scale))))

        highlight = None
        accepted = False
        skip = False

        while True:
            canvas = build_grid_image(crops, labels, highlight)
            info = f"Frame {g_idx+1}/{len(group_list)}  {ts}  — Enter:accept  s:skip  q:quit"
            cv2.putText(canvas, info, (4, ROWS * CELL - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.imshow("Grid labels", canvas)

            key = cv2.waitKey(30) & 0xFF

            # Handle pending mouse click
            if clicked_cell is not None:
                cell = clicked_cell
                clicked_cell = None
                if cell in labels:
                    cur = labels[cell]
                    labels[cell] = LABEL_CYCLE[(LABEL_CYCLE.index(cur) + 1) % len(LABEL_CYCLE)]
                    highlight = cell

            if key == ord('q'):
                cv2.destroyAllWindows()
                return

            if key == ord('s'):
                skip = True
                break

            if key in (13, ord('\n')):  # Enter
                accepted = True
                break

        if accepted:
            for (row, col), label in labels.items():
                src = path_map.get((row, col))
                if src is None or not src.exists():
                    continue
                dst = LABEL_DIRS[label] / src.name
                shutil.move(str(src), str(dst))
            print(f"[{ts}] saved {len(labels)} crops")
            group_list.pop(g_idx)  # don't advance; next group slides in
        elif skip:
            g_idx += 1

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
