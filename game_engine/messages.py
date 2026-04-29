from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from connect_4.config import BOARD
from .board import Connect4Board


class ControllerStatus(str, Enum):
    WAITING_FOR_BOARD = "waiting_for_board"
    HUMAN_TURN = "human_turn"
    ROBOT_DECIDING = "robot_deciding"
    ROBOT_MOVING = "robot_moving"
    GAME_OVER = "game_over"
    ERROR = "error"


@dataclass(frozen=True)
class MoveDecision:
    column: int
    reason: str = ""


@dataclass
class OrchestratorState:
    board: Connect4Board = field(default_factory=Connect4Board.empty)
    status: ControllerStatus = ControllerStatus.WAITING_FOR_BOARD
    last_human_col: Optional[int] = None
    last_robot_col: Optional[int] = None
    robot_target_col: Optional[int] = None
    winner: Optional[str] = None
    board_version: int = 0
    last_update_time: float = field(default_factory=time.time)
    last_error: Optional[str] = None
    move_history: List[str] = field(default_factory=list)
    awaiting_robot_confirmation: bool = False


class VisionBoardUpdate(BaseModel):
    board: List[List[str]] = Field(..., min_length=BOARD.rows, max_length=BOARD.rows)
    source: str = "vision_service"


class StatusResponse(BaseModel):
    status: str
    winner: Optional[str]
    last_human_col: Optional[int]
    last_robot_col: Optional[int]
    robot_target_col: Optional[int]
    board_version: int
    board_top_down: List[List[str]]
    pretty: str
    last_error: Optional[str]
    awaiting_robot_confirmation: bool
    move_history: List[str]
    motor_state: dict


class RobotMoveResponse(BaseModel):
    accepted: bool
    chosen_column: Optional[int] = None
    reason: Optional[str] = None
    status: str = "ok"
