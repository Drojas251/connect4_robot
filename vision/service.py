"""
Vision service — detects the Connect4 board state from camera.
Publishes board updates to orchestrator and web UI.

Startup sequence
----------------
1. searching        – scanning frames for AprilTag corners
2. checking         – corners found; classifying cells to verify board is empty
3. board_not_empty  – pieces detected; waiting for user to clear & retry
4. detection_failed – corners never found; waiting for retry
5. ready            – board empty, corners locked; main polling loop running
"""
from __future__ import annotations

import time
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from connect4_robot.config import SERVICES
from connect4_robot.game_engine.board import Connect4Board, Cell

try:
    from .connect4_tag_grid import Connect4TagGridDetector
    from .piece_color_classifier import PieceColorClassifier
    from .learned_piece_classifier import LearnedPieceClassifier
except ImportError:
    from connect4_tag_grid import Connect4TagGridDetector
    from piece_color_classifier import PieceColorClassifier
    from learned_piece_classifier import LearnedPieceClassifier


CAMERA_INDEX = 1
MAX_GRID_SEARCH_FRAMES = 300
REFRESH_SECONDS = 5
HOLE_CROP_RADIUS = 18
POLL_PERIOD_S = 0.1


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class VisionServiceState:
    def __init__(self):
        self.board = Connect4Board.empty()
        self.last_published_board = None
        self.is_running = False
        self.is_paused = False
        self.last_error: Optional[str] = None
        self.frame_count = 0
        self.last_detection_time = 0.0

        # Detection-phase fields
        self.detection_phase: str = "searching"
        self.detection_message: str = "Starting up…"
        self.detected_holes_count: int = 0
        self.detection_image_bytes: Optional[bytes] = None
        self.restart_requested: bool = False

        self._lock = threading.Lock()

    # --- board ---

    def update_board(self, board: Connect4Board):
        with self._lock:
            self.board = board
            self.last_detection_time = time.time()

    # --- detection phase ---

    def set_detection(self, phase: str, message: str):
        with self._lock:
            self.detection_phase = phase
            self.detection_message = message
        print(f"[vision] [{phase}] {message}")

    def set_detection_image(self, jpeg_bytes: bytes):
        with self._lock:
            self.detection_image_bytes = jpeg_bytes

    def request_restart(self):
        with self._lock:
            self.restart_requested = True
            self.board = Connect4Board.empty()
            self.last_published_board = None
            self.last_error = None
            self.frame_count = 0
            self.detection_image_bytes = None

    def set_paused(self, paused: bool):
        with self._lock:
            self.is_paused = paused

    def set_error(self, error: str):
        with self._lock:
            self.last_error = error

    def get_status(self) -> dict:
        with self._lock:
            return {
                "board_top_down": self.board.to_strings_top_down(),
                "is_paused": self.is_paused,
                "last_error": self.last_error,
                "frame_count": self.frame_count,
                "last_detection_time": self.last_detection_time,
                "detection_phase": self.detection_phase,
                "detection_message": self.detection_message,
                "detected_holes_count": self.detected_holes_count,
                "detection_image_available": self.detection_image_bytes is not None,
            }


vision_state = VisionServiceState()


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------

def circular_crop(frame, center_xy, radius):
    cx, cy = center_xy
    h, w = frame.shape[:2]
    x0, x1 = max(0, cx - radius), min(w, cx + radius)
    y0, y1 = max(0, cy - radius), min(h, cy + radius)
    patch = frame[y0:y1, x0:x1].copy()
    if patch.size == 0:
        return None
    mask = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), radius, 255, -1)
    return cv2.bitwise_and(patch, patch, mask=mask)


def classify_board(frame, holes, classifier) -> Connect4Board:
    board = Connect4Board.empty()
    for hole in holes:
        crop = circular_crop(frame, hole.frame_xy, HOLE_CROP_RADIUS)
        board.grid[hole.row][hole.col] = (
            classifier.classify(crop) if crop is not None else Cell.EMPTY
        )
    return board


def board_is_empty(board: Connect4Board) -> bool:
    return all(
        board.grid[r][c] == Cell.EMPTY
        for r in range(len(board.grid))
        for c in range(len(board.grid[0]))
    )


def render_detection_image(frame, holes, bbox_xyxy, board: Optional[Connect4Board] = None) -> bytes:
    """Draw the detected grid overlay and return JPEG bytes."""
    img = frame.copy()
    x0, y0, x1, y1 = bbox_xyxy
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 2)

    for hole in holes:
        cell = board.grid[hole.row][hole.col] if board else None
        if cell == Cell.HUMAN:
            color = (40, 210, 210)   # yellow (BGR)
        elif cell == Cell.ROBOT:
            color = (50, 50, 220)    # red (BGR)
        else:
            color = (180, 180, 180)  # grey for empty / unknown

        cv2.circle(img, hole.frame_xy, HOLE_CROP_RADIUS, color, 2)
        cv2.circle(img, hole.frame_xy, 4, color, -1)
        cv2.putText(
            img,
            f"r{hole.row}c{hole.col}",
            (hole.frame_xy[0] + 7, hole.frame_xy[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
        )

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return bytes(buf) if ok else b""


def find_grid(cap, detector, max_frames=MAX_GRID_SEARCH_FRAMES):
    """Scan frames until AprilTag grid is found.
    Exits early if vision_state.restart_requested is set.
    Always returns (result_or_None, last_frame_or_None)."""
    last_frame = None
    for i in range(max_frames):
        if not vision_state.is_running or vision_state.restart_requested:
            return None, last_frame
        ret, frame = cap.read()
        if not ret:
            continue
        last_frame = frame
        result = detector.detect_from_frame(frame)
        if result is not None and len(result.holes) > 0:
            print(f"[vision] Grid found on frame {i} ({len(result.holes)} holes)")
            return result, frame
        if i % 30 == 0 and i > 0:
            print(f"[vision] Searching… frame {i}")
    return None, last_frame


def publish_board_state(board: Connect4Board):
    payload = {"board": board.to_strings_top_down(), "source": "vision_service"}
    for name, url in [
        ("orchestrator", f"{SERVICES.orchestrator_url}/vision/update"),
        ("web",          f"{SERVICES.web_url}/api/board_update"),
    ]:
        try:
            requests.post(url, json=payload, timeout=5).raise_for_status()
        except Exception as e:
            print(f"[vision] Failed to publish to {name}: {e}")


# ---------------------------------------------------------------------------
# Main polling loop (runs in background thread)
# ---------------------------------------------------------------------------

def _wait_for_restart():
    while not vision_state.restart_requested and vision_state.is_running:
        time.sleep(0.4)


def vision_polling_loop(grid_detector, classifier):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        vision_state.set_detection("detection_failed", f"Cannot open camera {CAMERA_INDEX}")
        return

    try:
        while vision_state.is_running:
            vision_state.restart_requested = False

            # ── Phase 1: find AprilTag grid ───────────────────────────────
            vision_state.set_detection("searching", "Searching for board corners…")
            grid_result, grid_frame = find_grid(cap, grid_detector)

            if grid_result is None:
                # Store raw camera view so the user can see what the robot sees
                if grid_frame is not None:
                    ok, buf = cv2.imencode(".jpg", grid_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        vision_state.set_detection_image(bytes(buf))
                vision_state.set_detection(
                    "detection_failed",
                    "Board corners not detected. "
                    "Run 'python -m connect4_robot.vision.detect_board_corners' to debug.",
                )
                _wait_for_restart()
                continue

            holes = grid_result.holes
            bbox  = grid_result.bbox_xyxy
            vision_state.detected_holes_count = len(holes)

            # ── Phase 2: classify & verify empty ─────────────────────────
            vision_state.set_detection(
                "checking",
                f"Found {len(holes)} holes. Checking board is empty…",
            )

            ret, frame = cap.read()
            if not ret:
                vision_state.set_detection("detection_failed", "Camera read failed after grid lock.")
                _wait_for_restart()
                continue

            board = classify_board(frame, holes, classifier)
            vision_state.set_detection_image(render_detection_image(frame, holes, bbox, board))

            if not board_is_empty(board):
                occupied = sum(
                    1 for r in board.grid for c in r if c != Cell.EMPTY
                )
                vision_state.set_detection(
                    "board_not_empty",
                    f"{occupied} piece(s) detected. Clear all pieces from the board, then click Retry.",
                )
                vision_state.update_board(board)
                _wait_for_restart()
                continue

            # ── Phase 3: ready ────────────────────────────────────────────
            ret2, frame2 = cap.read()
            if ret2:
                vision_state.set_detection_image(render_detection_image(frame2, holes, bbox))

            vision_state.set_detection("ready", "Board detected and empty — game ready!")
            vision_state.update_board(Connect4Board.empty())

            # ── Phase 4: main polling loop ────────────────────────────────
            last_publish_time = 0.0

            while vision_state.is_running and not vision_state.restart_requested:
                if vision_state.is_paused:
                    time.sleep(POLL_PERIOD_S)
                    continue

                ret, frame = cap.read()
                if not ret:
                    time.sleep(POLL_PERIOD_S)
                    continue

                vision_state.frame_count += 1
                board = classify_board(frame, holes, classifier)
                vision_state.update_board(board)

                now = time.time()
                if now - last_publish_time >= REFRESH_SECONDS:
                    top_down = board.to_strings_top_down()
                    if vision_state.last_published_board != top_down:
                        print("[vision] Board changed — publishing")
                        publish_board_state(board)
                        vision_state.last_published_board = top_down
                    last_publish_time = now

                time.sleep(POLL_PERIOD_S)

    except Exception as e:
        vision_state.set_detection("detection_failed", f"Unexpected error: {e}")
        print(f"[vision] Fatal error: {e}")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

vision_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vision_thread
    try:
        grid_detector = Connect4TagGridDetector()
        try:
            classifier = LearnedPieceClassifier()
            print("[vision] Using LearnedPieceClassifier")
        except Exception as e:
            print(f"[vision] LearnedPieceClassifier unavailable ({e}), using PieceColorClassifier")
            classifier = PieceColorClassifier()

        vision_state.is_running = True
        vision_thread = threading.Thread(
            target=vision_polling_loop,
            args=(grid_detector, classifier),
            daemon=True,
        )
        vision_thread.start()
        print("[vision] Vision service started")
        yield
    except Exception as e:
        print(f"[vision] Startup error: {e}")
        vision_state.set_error(str(e))
        yield
    finally:
        vision_state.is_running = False
        if vision_thread is not None:
            vision_thread.join(timeout=5)
        print("[vision] Vision service stopped")


app = FastAPI(title="Connect4 Vision Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "service": "connect4-vision"}


@app.get("/status")
def get_status():
    return vision_state.get_status()


@app.get("/api/detection_image")
def detection_image():
    with vision_state._lock:
        data = vision_state.detection_image_bytes
    if not data:
        raise HTTPException(status_code=404, detail="No detection image yet")
    return Response(content=data, media_type="image/jpeg")


@app.post("/pause")
def pause():
    vision_state.set_paused(True)
    return {"ok": True, "paused": True}


@app.post("/resume")
def resume():
    vision_state.set_paused(False)
    return {"ok": True, "paused": False}


@app.post("/reset")
def reset():
    """Trigger a full re-detection cycle."""
    vision_state.request_restart()
    return {"ok": True, "restarting": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICES.vision_host, port=SERVICES.vision_port, reload=False)
