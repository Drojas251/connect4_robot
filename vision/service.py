"""
Vision service that detects the Connect4 board state from camera feed.
Publishes board state updates to orchestrator and web UI.
"""
from __future__ import annotations

import asyncio
import time
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from connect4_robot.config import SERVICES
from connect4_robot.game_engine.board import Connect4Board, Cell

# Try to import vision modules from local directory
try:
    from .connect4_tag_grid import Connect4TagGridDetector
    from .piece_color_classifier import PieceColorClassifier
    from .learned_piece_classifier import LearnedPieceClassifier
except ImportError:
    # Fallback for running as script
    from connect4_tag_grid import Connect4TagGridDetector
    from piece_color_classifier import PieceColorClassifier
    from learned_piece_classifier import LearnedPieceClassifier


# Configuration
CAMERA_INDEX = 1
MAX_GRID_SEARCH_FRAMES = 200
REFRESH_SECONDS = 5
HOLE_CROP_RADIUS = 18
POLL_PERIOD_S = 0.1


class BoardStateUpdate(BaseModel):
    """Board state published by vision service."""
    board_top_down: list[list[str]]
    timestamp: float
    source: str = "vision_service"


class VisionServiceState:
    """Internal state of the vision service."""
    
    def __init__(self):
        self.board = Connect4Board.empty()
        self.last_published_board = None
        self.is_running = False
        self.is_paused = False
        self.last_error = None
        self.frame_count = 0
        self.last_detection_time = 0.0
        self._lock = threading.Lock()
    
    def update_board(self, board: Connect4Board):
        """Update the current board state."""
        with self._lock:
            self.board = board
            self.last_detection_time = time.time()
    
    def get_board_state(self) -> dict:
        """Get current board state as a dict."""
        with self._lock:
            return {
                "board_top_down": self.board.to_strings_top_down(),
                "is_paused": self.is_paused,
                "last_error": self.last_error,
                "frame_count": self.frame_count,
                "last_detection_time": self.last_detection_time,
            }
    
    def set_paused(self, paused: bool):
        """Set pause state."""
        with self._lock:
            self.is_paused = paused
    
    def set_error(self, error: str):
        """Set error message."""
        with self._lock:
            self.last_error = error


vision_state = VisionServiceState()


def circular_crop_from_frame(frame, center_xy, radius):
    """Extract a circular patch from a frame."""
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
    local_cx = cx - x_min
    local_cy = cy - y_min

    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)

    return cv2.bitwise_and(patch, patch, mask=mask)


def find_grid_once(cap, grid_detector, max_frames=MAX_GRID_SEARCH_FRAMES):
    """Search for the Connect4 grid in video stream (headless — no display required)."""
    print(f"[vision] Searching up to {max_frames} frames for fixed grid...")

    for frame_idx in range(max_frames):
        ret, frame = cap.read()

        if not ret:
            print("[vision] Failed to read frame.")
            continue

        result = grid_detector.detect_from_frame(frame)

        if result is not None and len(result.holes) > 0:
            print(f"[vision] Grid found on frame {frame_idx}.")
            return result, frame.copy()

        if frame_idx % 30 == 0 and frame_idx > 0:
            print(f"[vision] Frame {frame_idx}: grid not found yet…")

    print("[vision] Grid not found after exhausting search frames.")
    return None, None


def classify_board_from_fixed_holes(frame, holes, classifier):
    """Classify pieces in detected holes."""
    board = Connect4Board.empty()

    for hole in holes:
        crop = circular_crop_from_frame(frame, hole.frame_xy, HOLE_CROP_RADIUS)
        cell = classifier.classify(crop) if crop is not None else Cell.EMPTY

        board.grid[hole.row][hole.col] = cell

    return board


def publish_board_state(board: Connect4Board):
    """Publish board state to orchestrator and web services."""
    try:
        payload = {
            "board": board.to_strings_top_down(),
            "source": "vision_service",
        }
        
        # Send to orchestrator
        try:
            resp = requests.post(
                f"{SERVICES.orchestrator_url}/vision/update",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[vision] Failed to publish to orchestrator: {e}")
        
        # Send to web UI
        try:
            resp = requests.post(
                f"{SERVICES.web_url}/api/board_update",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[vision] Failed to publish to web UI: {e}")
            
    except Exception as e:
        print(f"[vision] Error publishing board state: {e}")
        vision_state.set_error(str(e))


def vision_polling_loop(grid_detector, classifier):
    """Main vision polling loop that runs in a background thread."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        error = f"[vision] Failed to open camera {CAMERA_INDEX}"
        print(error)
        vision_state.set_error(error)
        return
    
    try:
        # Find grid once at startup
        grid_result, initial_frame = find_grid_once(cap, grid_detector)
        
        if grid_result is None or len(grid_result.holes) == 0:
            error = "[vision] Failed to find grid in camera feed"
            print(error)
            vision_state.set_error(error)
            return
        
        holes = grid_result.holes
        print(f"[vision] Found {len(holes)} holes")
        cv2.destroyAllWindows()
        
        # Main polling loop
        last_publish_time = 0.0
        
        while vision_state.is_running:
            if vision_state.is_paused:
                time.sleep(POLL_PERIOD_S)
                continue
            
            ret, frame = cap.read()
            if not ret:
                print("[vision] Failed to read frame")
                time.sleep(POLL_PERIOD_S)
                continue
            
            vision_state.frame_count += 1
            
            # Classify board from holes
            board = classify_board_from_fixed_holes(frame, holes, classifier)
            vision_state.update_board(board)
            
            # Publish periodically
            current_time = time.time()
            if current_time - last_publish_time >= REFRESH_SECONDS:
                board_top_down = board.to_strings_top_down()
                if vision_state.last_published_board != board_top_down:
                    print(f"[vision] Board changed, publishing update")
                    publish_board_state(board)
                    vision_state.last_published_board = board_top_down
                last_publish_time = current_time
            
            time.sleep(POLL_PERIOD_S)
    
    except Exception as e:
        error = f"[vision] Error in polling loop: {e}"
        print(error)
        vision_state.set_error(error)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


vision_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup and shutdown."""
    global vision_thread
    
    # Initialize vision components
    try:
        grid_detector = Connect4TagGridDetector()
        
        # Try LearnedPieceClassifier first, fall back to PieceColorClassifier
        try:
            classifier = LearnedPieceClassifier()
            print("[vision] Using LearnedPieceClassifier")
        except Exception as e:
            print(f"[vision] LearnedPieceClassifier failed ({e}), using PieceColorClassifier")
            classifier = PieceColorClassifier()
        
        # Start vision polling thread
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
        # Shutdown
        vision_state.is_running = False
        if vision_thread is not None:
            vision_thread.join(timeout=5)
        print("[vision] Vision service stopped")


app = FastAPI(
    title="Connect4 Vision Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "service": "connect4-vision"}


@app.get("/status")
def get_status():
    """Get current vision service status."""
    return vision_state.get_board_state()


@app.post("/pause")
def pause():
    """Pause vision polling."""
    vision_state.set_paused(True)
    return {"ok": True, "paused": True}


@app.post("/resume")
def resume():
    """Resume vision polling."""
    vision_state.set_paused(False)
    return {"ok": True, "paused": False}


@app.post("/reset")
def reset():
    """Reset vision service."""
    vision_state.board = Connect4Board.empty()
    vision_state.last_published_board = None
    vision_state.last_error = None
    vision_state.frame_count = 0
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVICES.vision_host,
        port=SERVICES.vision_port,
        reload=False,
    )
