#!/usr/bin/env python3
"""
detect_circle_grid.py — Connect4 circle-grid detector with Kalman smoothing.

Pipeline per frame:
  1. HSV-mask the blue plastic → board bounding box
  2. HoughCircles within bbox → raw detections
  3. Cluster detections into rows/cols; estimate regular grid spacing
  4. Fill all 42 expected slots (direct detection OR grid interpolation)
  5. Per-slot Kalman filter: low measurement noise for direct hits,
     high noise for interpolated positions
  6. Display in two windows: main overlay + blue-mask diagnostic

Run from the repo root:
    python vision/detect_circle_grid.py
    python vision/detect_circle_grid.py --image /tmp/board_capture.jpg

Controls (both modes):
    s       save debug frames to /tmp/circle_grid_*.jpg
    q/ESC   quit
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Point Qt at system fonts before cv2 loads its Qt backend.
# Without this the Qt window manager silently fails and shows blank windows.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import cv2
import numpy as np

# ── Tunable parameters ─────────────────────────────────────────────────────
CAMERA_INDEX  = 1
EXPECTED_ROWS = 6
EXPECTED_COLS = 7

# Blue board HSV range — widen hue if detection is unstable under your lighting
BLUE_H = (95, 135)
BLUE_S = (70, 255)
BLUE_V = (40, 255)
BLUE_MORPH_K = 11          # kernel size for morphological cleanup of mask

# HoughCircles — the main knobs to tune
HC_DP       = 1
HC_MIN_DIST = 55           # min px between circle centres
HC_P1       = 80           # Canny high threshold
HC_P2       = 20           # accumulator threshold (lower → more detections)
HC_MIN_R    = 24           # min hole radius (px)
HC_MAX_R    = 46           # max hole radius (px)

# Grid clustering tolerances (px) — increase if Hough centres scatter widely
ROW_TOL = 22
COL_TOL = 28

# Kalman filter
KF_Q         = 0.5    # process noise variance per frame while converging
KF_Q_SETTLED = 0.005  # process noise once a slot has settled — nearly frozen
KF_SETTLE_STD = 2.0   # std_px threshold to declare a slot settled (px)
KF_R_DIRECT  = 4.0    # measurement variance for a direct HoughCircles hit (~2 px std)
KF_R_INFERRED = 400.0 # measurement variance for a grid-interpolated position (~20 px std)
KF_INIT_COV  = 400.0  # initial P for a brand-new slot
# Distance gating: direct measurements beyond this many px from the KF estimate
# have their R scaled up as R_DIRECT * (dist/KF_GATE_R)^2, capped at R_INFERRED.
# Set equal to ~half a grid step so a one-slot Hough error gets heavily discounted.
KF_GATE_R = 20.0
# ──────────────────────────────────────────────────────────────────────────


# ── Per-slot Kalman filter ─────────────────────────────────────────────────

class _HoleKF:
    """2-state [cx, cy] Kalman filter for one static board hole."""

    def __init__(self, pos: tuple[float, float], direct: bool):
        self.x = np.array(pos, dtype=float)
        # Start confident if directly detected, uncertain if interpolated
        self.P = np.eye(2) * (KF_R_DIRECT if direct else KF_INIT_COV)
        self._settled = False

    def predict(self):
        # Once settled, inject almost no process noise — board doesn't move.
        q = KF_Q_SETTLED if self._settled else KF_Q
        self.P += q * np.eye(2)

    def update(self, z: tuple[float, float], direct: bool,
               expected: Optional[tuple[float, float]] = None):
        if direct:
            z_arr = np.asarray(z, float)
            dist_kf = float(np.linalg.norm(z_arr - self.x))
            if expected is not None:
                dist_grid = float(np.linalg.norm(z_arr - np.asarray(expected, float)))
                dist = min(dist_kf, dist_grid)
            else:
                dist = dist_kf
            # Scale R up quadratically with distance beyond KF_GATE_R.
            # A detection within KF_GATE_R of EITHER the KF estimate or the
            # geometric grid center gets full trust; further away → higher R.
            scale = max(1.0, (dist / KF_GATE_R) ** 2)
            R = min(KF_R_DIRECT * scale, KF_R_INFERRED)
        else:
            R = KF_R_INFERRED
        K = self.P @ np.linalg.inv(self.P + R * np.eye(2))
        self.x = self.x + K @ (np.asarray(z, float) - self.x)
        self.P = (np.eye(2) - K) @ self.P
        # Mark settled once uncertainty is small — future process noise is minimal.
        if not self._settled and self.std_px < KF_SETTLE_STD:
            self._settled = True

    @property
    def pos_int(self) -> tuple[int, int]:
        return (int(round(self.x[0])), int(round(self.x[1])))

    @property
    def std_px(self) -> float:
        return float(np.sqrt(np.mean(np.diag(self.P))))


class GridKalman:
    """One KF per board slot; handles predict/update bookkeeping."""

    def __init__(self):
        self._kfs: dict[tuple[int, int], _HoleKF] = {}

    # measurements: {(ri,ci): ((cx,cy), is_direct)}
    def step(
        self,
        measurements: dict[tuple[int, int], tuple[tuple[float, float], bool]],
        expected_centers: Optional[dict[tuple[int, int], tuple[int, int]]] = None,
    ) -> dict[tuple[int, int], tuple[tuple[int, int], float]]:
        """
        Returns {(ri,ci): (smoothed_pos, std_px)} for every known slot.
        expected_centers: geometric grid centers — used as a second reference
        point for distance-gating so a detection close to the grid ideal is
        trusted even when the KF hasn't converged yet.
        """
        seen: set[tuple[int, int]] = set()

        for slot, (pos, direct) in measurements.items():
            seen.add(slot)
            expected = expected_centers.get(slot) if expected_centers else None
            if slot not in self._kfs:
                self._kfs[slot] = _HoleKF(pos, direct)
            else:
                self._kfs[slot].predict()
                self._kfs[slot].update(pos, direct, expected=expected)

        # Advance time for slots with no measurement this frame
        for slot, kf in self._kfs.items():
            if slot not in seen:
                kf.predict()

        return {s: (kf.pos_int, kf.std_px) for s, kf in self._kfs.items()}


# ── Board detection ────────────────────────────────────────────────────────

def detect_blue_board(frame: np.ndarray):
    """Returns (bbox_xywh, blue_mask). bbox is None if no board found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo  = np.array([BLUE_H[0], BLUE_S[0], BLUE_V[0]])
    hi  = np.array([BLUE_H[1], BLUE_S[1], BLUE_V[1]])
    mask = cv2.inRange(hsv, lo, hi)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLUE_MORPH_K, BLUE_MORPH_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    biggest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 15_000:
        return None, mask
    return cv2.boundingRect(biggest), mask


# ── Circle detection ───────────────────────────────────────────────────────

def detect_circles(frame: np.ndarray, bbox=None) -> np.ndarray:
    """Returns Nx3 int32 array of (cx, cy, r) in full-frame coordinates."""
    if bbox is not None:
        x, y, w, h = bbox
        roi         = frame[y:y+h, x:x+w]
        ox, oy      = x, y
    else:
        roi    = frame
        ox, oy = 0, 0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    raw  = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=HC_DP, minDist=HC_MIN_DIST,
        param1=HC_P1, param2=HC_P2,
        minRadius=HC_MIN_R, maxRadius=HC_MAX_R,
    )
    if raw is None:
        return np.empty((0, 3), dtype=np.int32)

    circles = np.round(raw[0]).astype(np.int32)
    circles[:, 0] += ox
    circles[:, 1] += oy
    return circles


# ── Grid fitting ───────────────────────────────────────────────────────────

def _cluster_1d(values: np.ndarray, tol: float) -> list[list[int]]:
    order  = np.argsort(values)
    groups: list[list[int]] = [[int(order[0])]]
    for i in order[1:]:
        if values[i] - np.mean(values[groups[-1]]) < tol:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def _best_origin(values: np.ndarray, step: float, n_slots: int) -> float:
    """
    Find the grid origin that best aligns `values` to a regular grid with
    spacing `step` and `n_slots` slots (indices 0 … n_slots-1).

    We try every detected value as a possible anchor for every slot index,
    rather than anchoring only to the minimum.  This makes the estimate
    robust to a single outlier pulling min(values) off-grid — that outlier
    cannot win because assigning it to a valid slot will leave other circles
    outside [0, n_slots) and incur their 1.0 penalty each.
    """
    best_origin, best_score = float(values[0]), float("inf")
    for anchor in values:
        for slot_of_anchor in range(n_slots):
            origin = float(anchor) - slot_of_anchor * step
            err = 0.0
            for v in values:
                frac   = (v - origin) / step
                nearest = round(frac)
                if 0 <= nearest < n_slots:
                    err += (frac - nearest) ** 2
                else:
                    err += 1.0      # heavy penalty for out-of-bounds mapping
            score = err / len(values)
            if score < best_score:
                best_score  = score
                best_origin = origin
    return best_origin


def fit_grid(circles: np.ndarray):
    """
    Assign detected circles to absolute (row, col) grid slots.

    Key design: column indices are computed from round((x - origin_x) / dx)
    rather than from relative cluster ordering.  This means a missing circle
    at the left edge does NOT shift every other circle's column by one.

    Returns (assignments, grid_params) where:
      assignments : {circle_idx: (ri, ci)}
      grid_params : {dx, dy, origin, mean_r} or None
    """
    if len(circles) < 4:
        return {}, None

    ys = circles[:, 1].astype(float)
    xs = circles[:, 0].astype(float)

    # ── Step 1: cluster rows by y (rows are well-separated; this is robust) ──
    row_groups = sorted(_cluster_1d(ys, ROW_TOL), key=lambda g: np.mean(ys[g]))

    # ── Step 2: estimate dx using only single-step adjacent gaps ─────────────
    # Circle diameter ≈ 2 × HC_MAX_R; real grid spacing must exceed that.
    # Gaps from skipped circles are ≈ 2×dx, 3×dx, … — filter them out by
    # keeping only gaps close to the smallest *valid* gap.
    MIN_STEP = HC_MIN_DIST            # Hough can't produce two circles closer than this
    x_gaps: list[float] = []
    for rg in row_groups:
        sx = sorted(xs[i] for i in rg)
        x_gaps.extend(b - a for a, b in zip(sx, sx[1:]))
    if not x_gaps:
        return {}, None
    x_gaps.sort()
    valid_x = [g for g in x_gaps if g >= MIN_STEP]
    if not valid_x:
        return {}, None
    dx = float(np.median([g for g in valid_x if g < valid_x[0] * 1.6]))

    # ── Step 3: estimate dy using only single-step row-to-row gaps ───────────
    row_ys = [float(np.mean(ys[g])) for g in row_groups]
    y_gaps = sorted(row_ys[i+1] - row_ys[i] for i in range(len(row_ys)-1))
    if not y_gaps:
        return {}, None
    valid_y = [g for g in y_gaps if g >= MIN_STEP]
    if not valid_y:
        return {}, None
    dy = float(np.median([g for g in valid_y if g < valid_y[0] * 1.6]))

    # ── Step 4: find the absolute grid origin (column 0 / row 0 position) ────
    # Search over all possible slot-of-first-detection to get absolute indices.
    ox = _best_origin(xs, dx, EXPECTED_COLS)
    oy = _best_origin(ys, dy, EXPECTED_ROWS)

    # ── Step 5: assign each detection to its absolute (ri, ci) ───────────────
    assignments: dict[int, tuple[int, int]] = {}
    for idx in range(len(circles)):
        ci = int(round((xs[idx] - ox) / dx))
        ri = int(round((ys[idx] - oy) / dy))
        if 0 <= ri < EXPECTED_ROWS and 0 <= ci < EXPECTED_COLS:
            assignments[idx] = (ri, ci)

    mean_r = float(np.mean(circles[:, 2]))
    return assignments, dict(dx=dx, dy=dy, origin=(ox, oy), mean_r=mean_r)


def all_expected_centers(gp: dict) -> dict[tuple[int, int], tuple[int, int]]:
    """Return the expected (cx, cy) for all EXPECTED_ROWS × EXPECTED_COLS slots."""
    ox, oy = gp["origin"]
    dx, dy = gp["dx"], gp["dy"]
    return {
        (ri, ci): (int(round(ox + ci*dx)), int(round(oy + ri*dy)))
        for ri in range(EXPECTED_ROWS)
        for ci in range(EXPECTED_COLS)
    }


def build_measurements(
    circles: np.ndarray,
    assignments: dict[int, tuple[int, int]],
    grid_params: Optional[dict],
) -> dict[tuple[int, int], tuple[tuple[float, float], bool]]:
    """
    Build the full per-slot measurement dict for the KF.
    Direct HoughCircles detections → is_direct=True.
    Grid-interpolated fill-ins       → is_direct=False.
    """
    meas: dict[tuple[int, int], tuple[tuple[float, float], bool]] = {}

    # Direct detections take priority
    for idx, (ri, ci) in assignments.items():
        meas[(ri, ci)] = ((float(circles[idx, 0]), float(circles[idx, 1])), True)

    # Fill remaining slots from grid estimate
    if grid_params is not None:
        for (ri, ci), (ex, ey) in all_expected_centers(grid_params).items():
            if (ri, ci) not in meas:
                meas[(ri, ci)] = ((float(ex), float(ey)), False)

    return meas


# ── Visualisation ──────────────────────────────────────────────────────────

# Colours
_GREEN  = (0, 220, 0)
_ORANGE = (0, 150, 255)
_CYAN   = (255, 220, 0)


def draw_main_overlay(
    frame: np.ndarray,
    kf_output: dict[tuple[int, int], tuple[tuple[int, int], float]],
    direct_slots: set[tuple[int, int]],
    circles: np.ndarray,
    bbox,
    grid_params: Optional[dict],
) -> np.ndarray:
    vis = frame.copy()

    # Board bounding box
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 2)

    mean_r = int(round(grid_params["mean_r"])) if grid_params else 30

    # KF-smoothed positions
    for (ri, ci), (pos, std) in kf_output.items():
        cx, cy = pos
        direct = (ri, ci) in direct_slots
        color  = _GREEN if direct else _ORANGE
        thick  = 2     if direct else 1
        cv2.circle(vis, (cx, cy), mean_r, color, thick)
        cv2.putText(vis, f"{ri},{ci}", (cx - 13, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Raw HoughCircles — small cyan filled dot to distinguish from KF ring
    for cx, cy, r in circles:
        cv2.circle(vis, (cx, cy), 5, _CYAN, -1)

    # Stats banner
    n_direct  = len(direct_slots)
    n_inferred = len(kf_output) - n_direct
    banner = (
        f"Direct: {n_direct}  Inferred: {n_inferred}  "
        f"KF total: {len(kf_output)}/{EXPECTED_ROWS*EXPECTED_COLS}"
    )
    if grid_params:
        banner += f"   dx={grid_params['dx']:.1f} dy={grid_params['dy']:.1f}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, banner, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)

    return vis


def draw_mask_view(blue_mask: np.ndarray, bbox) -> np.ndarray:
    vis = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(vis, "Blue mask", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return vis


# ── Logging ────────────────────────────────────────────────────────────────

import csv, time, math

LOG_PATH       = "/tmp/grid_log.csv"
JUMP_LOG_PATH  = "/tmp/grid_jumps.txt"
JUMP_THRESHOLD = 20   # px — flag any slot that moves more than this per frame

_LOG_FIELDS = [
    "frame", "t",
    "ri", "ci",
    "meas_x", "meas_y", "is_direct",
    "kf_x", "kf_y", "kf_std",
    "delta_kf",          # distance KF position moved since last frame
    "n_hough", "n_direct", "n_inferred",
    "dx", "dy",
]

def open_log():
    f = open(LOG_PATH, "w", newline="")
    w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
    w.writeheader()
    return f, w

def log_frame(
    writer, jlog,
    frame_idx, t,
    meas, kf_output, prev_kf,
    n_hough, n_direct, n_inferred,
    gp,
):
    dx = gp["dx"] if gp else float("nan")
    dy = gp["dy"] if gp else float("nan")

    for slot, (kf_pos, kf_std) in kf_output.items():
        ri, ci = slot
        m_pos, m_direct = meas.get(slot, ((float("nan"), float("nan")), False))

        # Distance KF moved since last frame
        delta = float("nan")
        if slot in prev_kf:
            px, py = prev_kf[slot][0]
            delta = math.hypot(kf_pos[0] - px, kf_pos[1] - py)

            if delta > JUMP_THRESHOLD:
                line = (
                    f"JUMP  frame={frame_idx:4d} t={t:.3f}s  "
                    f"slot=({ri},{ci})  "
                    f"prev=({px},{py})  now=({kf_pos[0]},{kf_pos[1]})  "
                    f"delta={delta:.1f}px  "
                    f"meas=({m_pos[0]:.0f},{m_pos[1]:.0f}) direct={m_direct}"
                )
                print(line)
                jlog.write(line + "\n")

        writer.writerow({
            "frame": frame_idx, "t": f"{t:.3f}",
            "ri": ri, "ci": ci,
            "meas_x": f"{m_pos[0]:.1f}", "meas_y": f"{m_pos[1]:.1f}",
            "is_direct": int(m_direct),
            "kf_x": kf_pos[0], "kf_y": kf_pos[1],
            "kf_std": f"{kf_std:.2f}",
            "delta_kf": f"{delta:.1f}" if not math.isnan(delta) else "",
            "n_hough": n_hough, "n_direct": n_direct, "n_inferred": n_inferred,
            "dx": f"{dx:.2f}", "dy": f"{dy:.2f}",
        })


# ── Main loop ──────────────────────────────────────────────────────────────

def run_loop(source, kf: GridKalman):
    is_still = isinstance(source, np.ndarray)

    log_f, log_w = open_log()
    jlog = open(JUMP_LOG_PATH, "w")
    print(f"Logging to {LOG_PATH}  (jumps > {JUMP_THRESHOLD}px → {JUMP_LOG_PATH})")

    frame_idx = 0
    t0        = time.time()
    prev_kf: dict = {}

    try:
        while True:
            frame = source.copy() if is_still else None

            if not is_still:
                ret, frame = source.read()
                if not ret:
                    continue

            bbox, blue_mask = detect_blue_board(frame)
            circles         = detect_circles(frame, bbox)
            assignments, gp = fit_grid(circles)
            meas            = build_measurements(circles, assignments, gp)
            expected        = all_expected_centers(gp) if gp else None
            kf_output       = kf.step(meas, expected_centers=expected)

            direct_slots = {slot for slot, (_, d) in meas.items() if d}
            n_direct   = len(direct_slots)
            n_inferred = len(kf_output) - n_direct

            log_frame(
                log_w, jlog,
                frame_idx, time.time() - t0,
                meas, kf_output, prev_kf,
                len(circles), n_direct, n_inferred,
                gp,
            )
            log_f.flush()
            jlog.flush()
            prev_kf = dict(kf_output)
            frame_idx += 1

            vis_main = draw_main_overlay(frame, kf_output, direct_slots, circles, bbox, gp)
            vis_mask = draw_mask_view(blue_mask, bbox)

            cv2.imshow("Circle Grid - main", vis_main)
            cv2.imshow("Circle Grid - blue mask", vis_mask)

            wait_ms = 30 if is_still else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                cv2.imwrite("/tmp/circle_grid_debug.jpg", vis_main)
                cv2.imwrite("/tmp/circle_grid_mask.jpg",  vis_mask)
                print(f"  frame={frame_idx}  Direct: {n_direct}  Inferred: {n_inferred}  KF: {len(kf_output)}")
                if gp:
                    print(f"  dx={gp['dx']:.1f}  dy={gp['dy']:.1f}  r≈{gp['mean_r']:.1f}  origin={gp['origin']}")
                print("Saved debug frames.")
    finally:
        log_f.close()
        jlog.close()
        print(f"\nLog written to {LOG_PATH}")
        print(f"Jumps written to {JUMP_LOG_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Connect4 circle-grid detector")
    ap.add_argument("--image", help="Still image path (skips live camera)")
    args = ap.parse_args()

    kf = GridKalman()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image: {args.image}")
        print(f"Still-image mode: {args.image}  (s: save  q/ESC: quit)")
        run_loop(frame, kf)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            sys.exit(f"Cannot open camera {CAMERA_INDEX}")
        print("Live mode  (s: save  q/ESC: quit)")
        try:
            run_loop(cap, kf)
        finally:
            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
