from __future__ import annotations

import serial
import threading
import time
from dataclasses import dataclass
from typing import Optional


class MotorTransport:
    """
    Lowest-level serial transport.

    Important:
    - Writes are allowed from callers
    - Reads should only be performed by the MotorController reader thread
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._write_lock = threading.Lock()
        time.sleep(2.0)  # allow Arduino reset

    def write_line(self, line: str):
        with self._write_lock:
            self.ser.write((line + "\n").encode("utf-8"))
            self.ser.flush()

    def read_line(self) -> Optional[str]:
        raw = self.ser.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").strip()

    def close(self):
        if self.ser.is_open:
            self.ser.close()


@dataclass
class MotorState:
    pos_deg: Optional[float] = None
    vel_deg_s: Optional[float] = None
    moving: bool = False
    ready: bool = False
    mode: Optional[str] = None
    last_line: Optional[str] = None
    last_update_time: float = 0.0
    last_ack: Optional[str] = None
    last_error: Optional[str] = None


@dataclass
class EndEffectorState:
    state: str = "UNKNOWN"
    last_line: Optional[str] = None
    last_update_time: float = 0.0


# Added: limit switch state dataclass
@dataclass
class LimitSwitchState:
    name: str
    pin: int
    raw: int
    active: bool
    last_event: Optional[str] = None
    last_update_time: float = 0.0


@dataclass
class LinearAxisCalibration:
    """
    Linear mapping:

        deg = deg_per_mm * mm + deg_offset
        mm  = (deg - deg_offset) / deg_per_mm
    """
    deg_per_mm: float
    deg_offset: float = 0.0

    def mm_to_deg(self, mm: float) -> float:
        return self.deg_per_mm * mm + self.deg_offset

    def deg_to_mm(self, deg: float) -> float:
        return (deg - self.deg_offset) / self.deg_per_mm

    @classmethod
    def from_two_points(cls, mm1: float, deg1: float, mm2: float, deg2: float):
        if mm2 == mm1:
            raise ValueError("Calibration mm points must differ.")
        deg_per_mm = (deg2 - deg1) / (mm2 - mm1)
        deg_offset = deg1 - deg_per_mm * mm1
        return cls(deg_per_mm=deg_per_mm, deg_offset=deg_offset)


class MotorController:
    """
    High-level motor protocol/controller in motor degrees.

    This class owns the only serial read loop.
    All incoming lines are parsed here and stored in state.

    Expected Arduino protocol:
        READY
        ACK <command>
        ERR <message>
        STATE <pos_deg> <vel_deg_s> <busy> <mode>
        DONE <pos_deg>
        LIMIT_STATE <name> <pin> <raw> <active>
        LIMIT_EVENT <name> <event> <pin> <raw> <active>
        EE_STATE <state>
    """

    def __init__(self, transport: MotorTransport):
        self.transport = transport

        self._state = MotorState()
        self._state_lock = threading.Lock()

        self._ee_state = EndEffectorState()
        self._ee_lock = threading.Lock()

        # Limit switch state management
        self._limit_states: dict[str, LimitSwitchState] = {}
        self._limit_lock = threading.Lock()

        self._done_event = threading.Event()
        self._ready_event = threading.Event()
        self._shutdown_event = threading.Event()

        self._waiting_for_motion_start = False
        self._saw_motion_since_command = False
        self._command_lock = threading.Lock()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._wait_for_ready(timeout=5.0)

    def move_abs_deg(self, target_deg: float, max_vel_deg_s: float, max_accel_deg_s2: float):
        with self._command_lock:
            self._done_event.clear()
            self._waiting_for_motion_start = True
            self._saw_motion_since_command = False

            with self._state_lock:
                self._state.moving = True
                self._state.last_error = None

            self.transport.write_line(
                f"MOVE_ABS_DEG {target_deg} {max_vel_deg_s} {max_accel_deg_s2}"
            )

    def jog_deg_s(self, vel_deg_s: float, max_accel_deg_s2: float):
        print(f"[MotorController] jog_deg_s: vel={vel_deg_s} accel={max_accel_deg_s2}")
        with self._command_lock:
            self._done_event.clear()
            self._waiting_for_motion_start = True
            self._saw_motion_since_command = False

            with self._state_lock:
                self._state.moving = True
                self._state.last_error = None

            self.transport.write_line(f"JOG_DEG_S {vel_deg_s} {max_accel_deg_s2}")

    def stop(self):
        self.transport.write_line("STOP")

    def get_state(self):
        self.transport.write_line("GET_STATE")

    # New: request limit states from firmware
    def get_limits(self):
        self.transport.write_line("GET_LIMITS")

    def set_zero(self):
        self.transport.write_line("SET_ZERO")

    # -------------------------
    # End effector commands
    # -------------------------
    def dispense(self):
        self.transport.write_line("DISPENSE")

    def reload(self):
        self.transport.write_line("RELOAD")

    def get_end_effector_state(self):
        self.transport.write_line("GET_EE_STATE")

    def wait_for_ee_state(self, target_state: str, timeout: float = 5.0) -> bool:
        start = time.time()

        while time.time() - start < timeout:
            if self.get_ee_state_copy().state == target_state:
                return True
            time.sleep(0.02)

        return False

    def wait_until_ee_ready(self, timeout: float = 5.0) -> bool:
        return self.wait_for_ee_state("READY", timeout=timeout)

    def dispense_and_wait(self, timeout: float = 5.0) -> bool:
        self.dispense()

        if not self.wait_for_ee_state("DISPENSING", timeout=timeout):
            print("Failed to start dispensing within timeout")
            return False

        return self.wait_until_ee_ready(timeout=timeout)

    def reload_and_wait(self, timeout: float = 5.0) -> bool:
        self.reload()

        if not self.wait_for_ee_state("RELOADING", timeout=timeout):
            print("Failed to start reloading within timeout")
            return False

        return self.wait_until_ee_ready(timeout=timeout)

    def wait_until_done(self, timeout: float = 10.0):
        ok = self._done_event.wait(timeout=timeout)
        if not ok:
            state = self.get_state_copy()
            raise TimeoutError(
                f"Timed out waiting for motion completion. "
                f"last_line={state.last_line}, pos_deg={state.pos_deg}, "
                f"moving={state.moving}, mode={state.mode}, last_error={state.last_error}"
            )

    def get_state_copy(self) -> MotorState:
        with self._state_lock:
            return MotorState(
                pos_deg=self._state.pos_deg,
                vel_deg_s=self._state.vel_deg_s,
                moving=self._state.moving,
                ready=self._state.ready,
                mode=self._state.mode,
                last_line=self._state.last_line,
                last_update_time=self._state.last_update_time,
                last_ack=self._state.last_ack,
                last_error=self._state.last_error,
            )

    def get_ee_state_copy(self) -> EndEffectorState:
        with self._ee_lock:
            return EndEffectorState(
                state=self._ee_state.state,
                last_line=self._ee_state.last_line,
                last_update_time=self._ee_state.last_update_time,
            )

    # New: return a copy of current limit switch states
    def get_limit_states_copy(self) -> dict[str, LimitSwitchState]:
        with self._limit_lock:
            return {
                name: LimitSwitchState(
                    name=s.name,
                    pin=s.pin,
                    raw=s.raw,
                    active=s.active,
                    last_event=s.last_event,
                    last_update_time=s.last_update_time,
                )
                for name, s in self._limit_states.items()
            }

    def shutdown(self):
        self._shutdown_event.set()
        self._reader_thread.join(timeout=1.0)
        self.transport.close()

    def _wait_for_ready(self, timeout: float = 5.0):
        ok = self._ready_event.wait(timeout=timeout)
        if not ok:
            raise TimeoutError("Did not receive READY from motor controller.")

    def _reader_loop(self):
        while not self._shutdown_event.is_set():
            line = self.transport.read_line()
            if not line:
                continue
            self._handle_line(line.strip())

    def _handle_line(self, line: str):
        now = time.time()

        with self._state_lock:
            self._state.last_line = line
            self._state.last_update_time = now

        if line == "READY":
            with self._state_lock:
                self._state.ready = True
            self._ready_event.set()
            return

        if line.startswith("ACK "):
            with self._state_lock:
                self._state.last_ack = line
            return

        if line.startswith("ERR "):
            with self._state_lock:
                self._state.last_error = line
                self._state.moving = False

            with self._command_lock:
                self._waiting_for_motion_start = False
                self._saw_motion_since_command = False

            self._done_event.set()
            return

        if line.startswith("EE_STATE "):
            ee_state = line.replace("EE_STATE ", "").strip()
            with self._ee_lock:
                self._ee_state.state = ee_state
                self._ee_state.last_line = line
                self._ee_state.last_update_time = now
            return

        # New: handle limit STATE and EVENT lines
        if line.startswith("LIMIT_STATE "):
            parsed = self._parse_limit_state_line(line)
            if parsed is None:
                # ignore malformed
                return
            with self._limit_lock:
                self._limit_states[parsed.name] = parsed
            return

        if line.startswith("LIMIT_EVENT "):
            parsed = self._parse_limit_event_line(line)
            if parsed is None:
                # ignore malformed
                return
            with self._limit_lock:
                self._limit_states[parsed.name] = parsed
            return

        if line.startswith("STATE "):
            parsed = self._parse_state_line(line)
            if parsed is None:
                print(f"[motor parse warning] Could not parse STATE line: {line}")
                return

            with self._state_lock:
                self._state.pos_deg = parsed["pos_deg"]
                self._state.vel_deg_s = parsed["vel_deg_s"]
                self._state.moving = parsed["moving"]
                self._state.mode = parsed["mode"]

            with self._command_lock:
                if parsed["moving"]:
                    self._saw_motion_since_command = True
                    self._waiting_for_motion_start = False
                else:
                    if self._saw_motion_since_command:
                        self._waiting_for_motion_start = False
                        self._done_event.set()

            return

        if line.startswith("DONE "):
            parsed = self._parse_done_line(line)
            if parsed is None:
                print(f"[motor parse warning] Could not parse DONE line: {line}")
                return

            with self._state_lock:
                self._state.pos_deg = parsed["pos_deg"]
                self._state.vel_deg_s = 0.0
                self._state.moving = False
                self._state.mode = "IDLE"

            with self._command_lock:
                self._waiting_for_motion_start = False
                self._saw_motion_since_command = False

            self._done_event.set()
            return

    def _parse_state_line(self, line: str) -> Optional[dict]:
        parts = line.split()
        if len(parts) != 5 or parts[0] != "STATE":
            return None

        try:
            pos_deg = float(parts[1])
            vel_deg_s = float(parts[2])
            moving = bool(int(parts[3]))
            mode = parts[4]
        except (ValueError, IndexError):
            return None

        return {
            "pos_deg": pos_deg,
            "vel_deg_s": vel_deg_s,
            "moving": moving,
            "mode": mode,
        }

    def _parse_done_line(self, line: str) -> Optional[dict]:
        parts = line.split()
        if len(parts) != 2 or parts[0] != "DONE":
            return None

        try:
            pos_deg = float(parts[1])
        except ValueError:
            return None

        return {"pos_deg": pos_deg}

    # New: parse LIMIT_STATE
    def _parse_limit_state_line(self, line: str) -> Optional[LimitSwitchState]:
        parts = line.split()
        if len(parts) != 5 or parts[0] != "LIMIT_STATE":
            return None
        try:
            return LimitSwitchState(
                name=parts[1],
                pin=int(parts[2]),
                raw=int(parts[3]),
                active=bool(int(parts[4])),
                last_event=None,
                last_update_time=time.time(),
            )
        except ValueError:
            return None

    # New: parse LIMIT_EVENT
    def _parse_limit_event_line(self, line: str) -> Optional[LimitSwitchState]:
        parts = line.split()
        if len(parts) != 6 or parts[0] != "LIMIT_EVENT":
            return None
        try:
            return LimitSwitchState(
                name=parts[1],
                pin=int(parts[3]),
                raw=int(parts[4]),
                active=bool(int(parts[5])),
                last_event=parts[2],
                last_update_time=time.time(),
            )
        except ValueError:
            return None


class LinearSliderAxis:
    """
    Motion interface in mm / mm/s.
    Uses MotorController internally, which remains in degrees.
    """

    def __init__(self, controller: MotorController, calibration: LinearAxisCalibration):
        self.controller = controller
        self.calibration = calibration

    def move_to_mm(self, target_mm: float, max_vel_mm_s: float, max_accel_mm_s2: float):
        target_deg = self.calibration.mm_to_deg(target_mm)
        vel_deg_s = abs(self.calibration.deg_per_mm) * max_vel_mm_s
        accel_deg_s2 = abs(self.calibration.deg_per_mm) * max_accel_mm_s2

        self.controller.move_abs_deg(
            target_deg=target_deg,
            max_vel_deg_s=vel_deg_s,
            max_accel_deg_s2=accel_deg_s2,
        )

    def jog_mm_s(self, vel_mm_s: float, max_accel_mm_s2: float):
        vel_deg_s = self.calibration.deg_per_mm * vel_mm_s
        accel_deg_s2 = abs(self.calibration.deg_per_mm) * max_accel_mm_s2

        self.controller.jog_deg_s(
            vel_deg_s=vel_deg_s,
            max_accel_deg_s2=accel_deg_s2,
        )

    def stop(self):
        self.controller.stop()

    def request_state(self):
        self.controller.get_state()

    # New: request limits
    def request_limits(self):
        self.controller.get_limits()

    def wait_until_done(self, timeout: float = 10.0):
        self.controller.wait_until_done(timeout=timeout)

    def get_position_deg(self) -> Optional[float]:
        state = self.controller.get_state_copy()
        return state.pos_deg

    def get_position_mm(self) -> Optional[float]:
        state = self.controller.get_state_copy()
        if state.pos_deg is None:
            return None
        return self.calibration.deg_to_mm(state.pos_deg)

    def get_velocity_mm_s(self) -> Optional[float]:
        state = self.controller.get_state_copy()
        if state.vel_deg_s is None:
            return None
        return state.vel_deg_s / abs(self.calibration.deg_per_mm)

    def set_zero_here(self, mm_value: float = 0.0):
        self.controller.set_zero()
        self.calibration.deg_offset = -self.calibration.deg_per_mm * mm_value

    # New: expose limit states as simple dicts
    def get_limit_states(self) -> dict:
        states = self.controller.get_limit_states_copy()
        return {
            name: {
                "name": s.name,
                "pin": s.pin,
                "raw": s.raw,
                "active": s.active,
                "last_event": s.last_event,
                "last_update_time": s.last_update_time,
            }
            for name, s in states.items()
        }

    def is_limit_active(self, name: str) -> bool:
        limits = self.get_limit_states()
        if name not in limits:
            return False
        return bool(limits[name]["active"])

    def wait_for_limit(
        self,
        name: str,
        *,
        active: bool = True,
        timeout: float = 10.0,
        poll_period: float = 0.02,
    ):
        t0 = time.time()

        while time.time() - t0 < timeout:
            self.request_limits()
            time.sleep(poll_period)

            limits = self.get_limit_states()
            if name not in limits:
                continue

            if bool(limits[name]["active"]) == active:
                return limits[name]

        raise TimeoutError(f"Timed out waiting for limit {name!r} active={active}.")

    def home_to_limit(
        self,
        limit_name: str = "home_min",
        *,
        direction: float = -1.0,
        home_speed_mm_s: float = 50.0,
        home_accel_mm_s2: float = 100.0,
        timeout: float = 15.0,
        zero_mm: float = 0.0,
        backoff_mm: float = 0.0,
        backoff_speed_mm_s: float = 10.0,
        poll_period: float = 0.02,
    ):
        """
        Homes the axis by jogging until a limit switch becomes active.

        Sequence:
          1. Request current limit state.
          2. Jog in direction at home_speed_mm_s.
          3. Poll limit state until active.
          4. Stop the axis.
          5. Set current position as zero_mm.
          6. Optionally back off by backoff_mm.
        """
        if direction == 0:
            raise ValueError("direction must be nonzero.")
        if home_speed_mm_s <= 0:
            raise ValueError("home_speed_mm_s must be positive.")
        if home_accel_mm_s2 <= 0:
            raise ValueError("home_accel_mm_s2 must be positive.")

        # If already on the switch, we can zero immediately.
        self.request_limits()
        time.sleep(poll_period)
        if self.is_limit_active(limit_name):
            print(f"[home] limit {limit_name} already active; zeroing.")
            self.stop()
            time.sleep(0.1)
            self.set_zero_here(zero_mm)
            return

        vel = abs(home_speed_mm_s) * (1.0 if direction > 0 else -1.0)
        print(f"[home] jogging at {vel} mm/s toward limit {limit_name}")

        self.jog_mm_s(vel, home_accel_mm_s2)

        try:
            limit_state = self.wait_for_limit(
                limit_name,
                active=True,
                timeout=timeout,
                poll_period=poll_period,
            )
            print(f"[home] hit limit: {limit_state}")
        except Exception:
            self.stop()
            raise

        self.stop()
        time.sleep(0.2)

        self.set_zero_here(zero_mm)
        print(f"[home] set current position to {zero_mm} mm")

        if backoff_mm != 0.0:
            # Move away from the switch. If homing direction was negative,
            # positive backoff moves away; if homing positive, negative backs off.
            backoff_target = zero_mm + (abs(backoff_mm) * (-1.0 if direction > 0 else 1.0))
            print(f"[home] backing off to {backoff_target} mm")
            self.move_to_mm(
                backoff_target,
                max_vel_mm_s=backoff_speed_mm_s,
                max_accel_mm_s2=home_accel_mm_s2,
            )
            self.wait_until_done(timeout=timeout)

    def get_state_dict(self) -> dict:
        motor_state = self.controller.get_state_copy()
        ee_state = self.controller.get_ee_state_copy()

        pos_mm = None
        if motor_state.pos_deg is not None:
            pos_mm = self.calibration.deg_to_mm(motor_state.pos_deg)

        vel_mm_s = None
        if motor_state.vel_deg_s is not None:
            vel_mm_s = motor_state.vel_deg_s / abs(self.calibration.deg_per_mm)

        return {
            "pos_deg": motor_state.pos_deg,
            "vel_deg_s": motor_state.vel_deg_s,
            "moving": motor_state.moving,
            "ready": motor_state.ready,
            "mode": motor_state.mode,
            "pos_mm": pos_mm,
            "vel_mm_s": vel_mm_s,
            "last_line": motor_state.last_line,
            "last_update_time": motor_state.last_update_time,
            "last_ack": motor_state.last_ack,
            "last_error": motor_state.last_error,
            "limits": self.get_limit_states(),
            "end_effector": {
                "state": ee_state.state,
                "last_line": ee_state.last_line,
                "last_update_time": ee_state.last_update_time,
            },
        }


class Connect4Gantry:
    """
    Application-level gantry controller for Connect 4.
    """

    def __init__(
        self,
        axis: LinearSliderAxis,
        column_centers_mm: list[float],
        home_mm: Optional[float] = None,
        staging_mm: Optional[float] = None,
        right_clearance_mm: Optional[float] = None,
    ):
        if len(column_centers_mm) < 1:
            raise ValueError("column_centers_mm must not be empty.")

        self.axis = axis
        self.column_centers_mm = column_centers_mm
        self.home_mm = home_mm
        self.staging_mm = staging_mm
        self.right_clearance_mm = right_clearance_mm

    def move_to_column(
        self,
        col_idx: int,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
    ):
        self._validate_col(col_idx)
        target_mm = self.column_centers_mm[col_idx]
        self.axis.move_to_mm(
            target_mm=target_mm,
            max_vel_mm_s=max_vel_mm_s,
            max_accel_mm_s2=max_accel_mm_s2,
        )

    def move_to_column_and_wait(
        self,
        col_idx: int,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
        timeout: float = 10.0,
    ):
        self.move_to_column(col_idx, max_vel_mm_s, max_accel_mm_s2)
        self.axis.wait_until_done(timeout=timeout)

    def move_home(
        self,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
    ):
        if self.home_mm is None:
            raise ValueError("home_mm is not configured.")
        self.axis.move_to_mm(self.home_mm, max_vel_mm_s, max_accel_mm_s2)

    def move_staging(
        self,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
    ):
        if self.staging_mm is None:
            raise ValueError("staging_mm is not configured.")
        self.axis.move_to_mm(self.staging_mm, max_vel_mm_s, max_accel_mm_s2)

    def get_column_center_mm(self, col_idx: int) -> float:
        self._validate_col(col_idx)
        return self.column_centers_mm[col_idx]

    def get_current_mm(self) -> Optional[float]:
        return self.axis.get_position_mm()

    def reload_piece(self, timeout: float = 5.0):
        print("[connect4] reloading end effector")

        ok = self.axis.controller.reload_and_wait(timeout=timeout)
        if not ok:
            raise TimeoutError("End effector failed to reload.")

        print("[connect4] reload complete")

    def drop_piece(self, timeout: float = 5.0):
        print("[connect4] dropping piece")

        ok = self.axis.controller.dispense_and_wait(timeout=timeout)
        if not ok:
            raise TimeoutError("End effector failed to dispense piece.")

        print("[connect4] piece dropped")

    def move_to_clearance(
        self,
        col_idx: int,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
        timeout: float = 10.0,
    ):
        """Move out of the player's way: go to the nearest board edge after a drop."""
        n = len(self.column_centers_mm)
        go_left = col_idx < n // 2

        if go_left and self.staging_mm is not None:
            target_mm = self.staging_mm
        elif not go_left and self.right_clearance_mm is not None:
            target_mm = self.right_clearance_mm
        else:
            return  # no clearance position configured for this side

        print(f"[connect4] clearing to {'left' if go_left else 'right'} ({target_mm} mm)")
        self.axis.move_to_mm(target_mm, max_vel_mm_s=max_vel_mm_s, max_accel_mm_s2=max_accel_mm_s2)
        self.axis.wait_until_done(timeout=timeout)

    def place_piece(
        self,
        col_idx: int,
        max_vel_mm_s: float = 80.0,
        max_accel_mm_s2: float = 300.0,
        timeout: float = 10.0,
        drop_timeout: float = 5.0,
    ):
        self.move_to_column_and_wait(
            col_idx=col_idx,
            max_vel_mm_s=max_vel_mm_s,
            max_accel_mm_s2=max_accel_mm_s2,
            timeout=timeout,
        )
        time.sleep(0.5)  # small delay before dropping
        self.drop_piece(timeout=drop_timeout)
        time.sleep(1.5)  # allow piece to settle
        self.reload_piece(timeout=drop_timeout)
        self.move_to_clearance(col_idx, max_vel_mm_s=max_vel_mm_s, max_accel_mm_s2=max_accel_mm_s2, timeout=timeout)

    def _validate_col(self, col_idx: int):
        if not (0 <= col_idx < 7):
            raise ValueError(f"Column index must be in [0, 6], got {col_idx}.")
