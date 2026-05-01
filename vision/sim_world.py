from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from connect4_robot.config import BOARD
from connect4_robot.game_engine.board import Cell


@dataclass
class SimBoard:
    grid: List[List[Cell]] = field(
        default_factory=lambda: [[Cell.EMPTY for _ in range(BOARD.cols)] for _ in range(BOARD.rows)]
    )

    def to_strings_top_down(self) -> List[List[str]]:
        return [[self.grid[r][c].value for c in range(BOARD.cols)] for r in reversed(range(BOARD.rows))]

    def can_play(self, col: int) -> bool:
        return 0 <= col < BOARD.cols and self.grid[BOARD.rows - 1][col] == Cell.EMPTY

    def legal_moves(self) -> List[int]:
        return [c for c in range(BOARD.cols) if self.can_play(c)]

    def apply_move(self, col: int, piece: Cell) -> int:
        for r in range(BOARD.rows):
            if self.grid[r][col] == Cell.EMPTY:
                self.grid[r][col] = piece
                return r
        raise ValueError(f"Column {col} full")


@dataclass
class VisionState:
    board: SimBoard = field(default_factory=SimBoard)
    last_sent_board: Optional[List[List[str]]] = None
    move_history: List[str] = field(default_factory=list)
