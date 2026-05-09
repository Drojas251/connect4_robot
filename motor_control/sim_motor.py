"""
Fake gantry for simulation mode.

Provides drop-in stand-ins for `LinearSliderAxis` and `Connect4Gantry` that
satisfy the surface area used by `Connect4Orchestrator` without touching any
hardware: no serial port, no Arduino, no homing limit switches.

All motion just sleeps for a short time and updates an in-memory position.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class SimAxis:
    """Stand-in for `LinearSliderAxis` — tracks a fake position in mm."""

    def __init__(self, start_mm: float = 0.0):
        self._lock = threading.Lock()
        self._pos_mm = start_mm
        self._moving = False
        self._mode = "IDLE"
        self._last_update = time.time()

    # -- motion --------------------------------------------------------

    def move_to_mm(
        self,
        target_mm: float,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
    ):
        with self._lock:
            distance = abs(target_mm - self._pos_mm)
            self._mode = "MOVING"
            self._moving = True
        # Approximate the move time with constant velocity; cap so tests stay
        # fast even when distances are large.
        dt = min(2.0, distance / max(1e-3, max_vel_mm_s))
        time.sleep(dt)
        with self._lock:
            self._pos_mm = target_mm
            self._moving = False
            self._mode = "IDLE"
            self._last_update = time.time()

    def home_to_limit(self, **_kwargs):
        # Pretend we drove to the home switch and zeroed.
        with self._lock:
            self._mode = "HOMING"
            self._moving = True
        time.sleep(0.3)
        with self._lock:
            self._pos_mm = 0.0
            self._moving = False
            self._mode = "IDLE"
            self._last_update = time.time()

    def wait_until_done(self, timeout: float = 10.0):
        # `move_to_mm` is synchronous in sim, so nothing to wait for.
        return

    # -- state for /status --------------------------------------------

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "pos_deg": None,
                "vel_deg_s": 0.0,
                "moving": self._moving,
                "ready": True,
                "mode": self._mode,
                "pos_mm": self._pos_mm,
                "vel_mm_s": 0.0,
                "last_line": "SIM",
                "last_update_time": self._last_update,
                "last_ack": None,
                "last_error": None,
                "limits": {},
                "end_effector": {
                    "state": "READY",
                    "last_line": "SIM",
                    "last_update_time": self._last_update,
                },
                "sim": True,
            }

    # -- shutdown surface used by Connect4Orchestrator ----------------

    class _SimController:
        def shutdown(self):
            return

    @property
    def controller(self) -> "_SimController":
        return SimAxis._SimController()


class SimGantry:
    """Stand-in for `Connect4Gantry` — implements `place_piece(col)` etc."""

    def __init__(
        self,
        column_centers_mm: list[float],
        home_mm: Optional[float] = None,
        staging_mm: Optional[float] = None,
        right_clearance_mm: Optional[float] = None,
    ):
        self.axis = SimAxis(start_mm=home_mm if home_mm is not None else 0.0)
        self.column_centers_mm = column_centers_mm
        self.home_mm = home_mm
        self.staging_mm = staging_mm
        self.right_clearance_mm = right_clearance_mm

    def place_piece(self, col_idx: int, timeout: float = 20.0, **_kwargs):
        if not (0 <= col_idx < len(self.column_centers_mm)):
            raise ValueError(f"col_idx {col_idx} out of range")
        target_mm = self.column_centers_mm[col_idx]
        print(f"[sim-gantry] place_piece col={col_idx} target_mm={target_mm}")
        self.axis.move_to_mm(target_mm)
        time.sleep(0.1)  # pretend we dropped + reloaded
        # Move to a clearance position so the (fake) human can see the board.
        n = len(self.column_centers_mm)
        go_left = col_idx < n // 2
        if go_left and self.staging_mm is not None:
            self.axis.move_to_mm(self.staging_mm)
        elif not go_left and self.right_clearance_mm is not None:
            self.axis.move_to_mm(self.right_clearance_mm)
        print(f"[sim-gantry] place_piece col={col_idx} done")
