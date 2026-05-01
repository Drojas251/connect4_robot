from __future__ import annotations

import threading
import time

from connect4_robot.config import BOARD, MOTOR
from connect4_robot.motor_control import (
    MotorTransport,
    MotorController,
    LinearAxisCalibration,
    LinearSliderAxis,
    Connect4Gantry,
)

from .board import board_from_top_down_strings
from .messages import (
    ControllerStatus,
    OrchestratorState,
    VisionBoardUpdate,
    StatusResponse,
    RobotMoveResponse,
)
from .policy import WeightedRandomPolicy


class Connect4Orchestrator:
    def __init__(self, gantry: Connect4Gantry, policy: WeightedRandomPolicy):
        self.gantry = gantry
        self.policy = policy
        self.state = OrchestratorState()
        self._lock = threading.RLock()

    def _append_history(self, msg: str):
        self.state.move_history.append(msg)
        self.state.move_history = self.state.move_history[-30:]

    def get_status(self) -> StatusResponse:
        with self._lock:
            return StatusResponse(
                status=self.state.status.value,
                winner=self.state.winner,
                last_human_col=self.state.last_human_col,
                last_robot_col=self.state.last_robot_col,
                robot_target_col=self.state.robot_target_col,
                board_version=self.state.board_version,
                board_top_down=self.state.board.to_strings_top_down(),
                pretty=self.state.board.pretty(),
                last_error=self.state.last_error,
                awaiting_robot_confirmation=self.state.awaiting_robot_confirmation,
                move_history=list(self.state.move_history),
                motor_state=self.gantry.axis.get_state_dict(),
            )

    def reset(self):
        with self._lock:
            try:
                self.gantry.axis.home_to_limit(
                    limit_name="home_min",
                    direction=-1.0,
                    home_speed_mm_s=20.0,
                    home_accel_mm_s2=100.0,
                    timeout=20.0,
                    zero_mm=0.0,
                    backoff_mm=5.0,
                    backoff_speed_mm_s=10.0,
                )
                self._append_history("homed gantry on reset")
            except Exception as e:
                self._append_history(f"homing failed on reset: {e}")
            self.state = OrchestratorState()
            self.state.status = ControllerStatus.HUMAN_TURN
            self.state.board_version = 1
            self._append_history("reset")

    def handle_vision_board_update(self, payload: VisionBoardUpdate) -> RobotMoveResponse:
        new_board = board_from_top_down_strings(payload.board)
        if not new_board.is_valid_physical_board():
            raise ValueError("Vision board is not physically valid.")

        with self._lock:
            old_board = self.state.board.copy()

            if self.state.board_version == 0:
                self.state.board_version = 1
                self.state.status = ControllerStatus.HUMAN_TURN
                self._append_history("initial orchestrator startup")

            if old_board.to_strings_top_down() == new_board.to_strings_top_down():
                self.state.board = new_board
                return RobotMoveResponse(accepted=True, reason="no_change_detected")

            inferred = old_board.infer_single_new_move(new_board)
            self.state.board = new_board
            self.state.board_version += 1
            self.state.last_update_time = time.time()

            winner_piece = new_board.check_winner()
            if winner_piece is not None:
                self.state.status = ControllerStatus.GAME_OVER
                self.state.winner = "human" if winner_piece.value == "H" else "robot"
                self._append_history(f"winner={self.state.winner}")
                return RobotMoveResponse(accepted=True, reason="game_over_detected")

            if new_board.is_full():
                self.state.status = ControllerStatus.GAME_OVER
                self._append_history("draw")
                return RobotMoveResponse(accepted=True, reason="board_full")

            if inferred is None:
                self._append_history("vision change without single inferred move")
                return RobotMoveResponse(accepted=True, reason="board_updated_no_single_new_move")

            _, col, piece = inferred

            if piece.value == "R":
                self.state.last_robot_col = col
                self.state.awaiting_robot_confirmation = False
                self.state.robot_target_col = None
                self.state.status = ControllerStatus.HUMAN_TURN
                self._append_history(f"vision confirmed robot in col {col}")
                return RobotMoveResponse(accepted=True, reason="robot_move_confirmed")

            self.state.last_human_col = col
            self.state.status = ControllerStatus.ROBOT_DECIDING
            self._append_history(f"vision sensed human in col {col}")

        return self._decide_and_execute_robot_move()

    def _decide_and_execute_robot_move(self) -> RobotMoveResponse:
        with self._lock:
            decision = self.policy.choose_move(self.state.board)
            self.state.status = ControllerStatus.ROBOT_MOVING
            self.state.robot_target_col = decision.column
            self.state.awaiting_robot_confirmation = True
            self._append_history(f"robot target col {decision.column}")

        self.gantry.place_piece(decision.column, timeout=20.0)

        with self._lock:
            self._append_history(
                f"gantry motion complete for col {decision.column}; waiting for vision confirmation"
            )

        return RobotMoveResponse(
            accepted=True,
            chosen_column=decision.column,
            reason=decision.reason,
        )

    def shutdown(self):
        try:
            self.gantry.axis.controller.shutdown()
        except Exception:
            pass


def build_orchestrator() -> Connect4Orchestrator:
    transport = MotorTransport(MOTOR.port, MOTOR.baudrate, MOTOR.timeout_s)
    controller = MotorController(transport)
    calibration = LinearAxisCalibration(deg_per_mm=MOTOR.deg_per_mm, deg_offset=MOTOR.deg_offset)
    axis = LinearSliderAxis(controller, calibration)
    gantry = Connect4Gantry(
        axis=axis,
        column_centers_mm=BOARD.column_centers_mm,
        home_mm=BOARD.home_mm,
        staging_mm=BOARD.staging_mm,
    )
    return Connect4Orchestrator(gantry=gantry, policy=WeightedRandomPolicy())
