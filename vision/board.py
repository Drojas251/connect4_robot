from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from config import BOARD


class Cell(str, Enum):
    EMPTY = "."
    HUMAN = "H"
    ROBOT = "R"


@dataclass
class Connect4Board:
    grid: List[List[Cell]] = field(
        default_factory=lambda: [[Cell.EMPTY for _ in range(BOARD.cols)] for _ in range(BOARD.rows)]
    )

    @classmethod
    def empty(cls) -> "Connect4Board":
        return cls()

    def copy(self) -> "Connect4Board":
        return Connect4Board(grid=[[cell for cell in row] for row in self.grid])

    def to_strings_top_down(self) -> List[List[str]]:
        return [[self.grid[r][c].value for c in range(BOARD.cols)] for r in reversed(range(BOARD.rows))]

    def pretty(self) -> str:
        lines = []
        for r in reversed(range(BOARD.rows)):
            lines.append(" ".join(self.grid[r][c].value for c in range(BOARD.cols)))
        lines.append(" ".join(str(c) for c in range(BOARD.cols)))
        return "\n".join(lines)

    def can_play(self, col: int) -> bool:
        return 0 <= col < BOARD.cols and self.grid[BOARD.rows - 1][col] == Cell.EMPTY

    def legal_moves(self) -> List[int]:
        return [c for c in range(BOARD.cols) if self.can_play(c)]

    def next_open_row(self, col: int) -> Optional[int]:
        for r in range(BOARD.rows):
            if self.grid[r][col] == Cell.EMPTY:
                return r
        return None

    def apply_move(self, col: int, piece: Cell) -> int:
        row = self.next_open_row(col)
        if row is None:
            raise ValueError(f"Column {col} is full.")
        self.grid[row][col] = piece
        return row

    def infer_single_new_move(self, new_board: "Connect4Board") -> Optional[Tuple[int, int, Cell]]:
        diffs = []
        for r in range(BOARD.rows):
            for c in range(BOARD.cols):
                old = self.grid[r][c]
                new = new_board.grid[r][c]
                if old != new:
                    diffs.append((r, c, old, new))
        if len(diffs) != 1:
            return None
        r, c, old, new = diffs[0]
        if old == Cell.EMPTY and new in (Cell.HUMAN, Cell.ROBOT):
            return (r, c, new)
        return None

    def is_valid_physical_board(self) -> bool:
        for c in range(BOARD.cols):
            seen_empty = False
            for r in range(BOARD.rows):
                if self.grid[r][c] == Cell.EMPTY:
                    seen_empty = True
                elif seen_empty:
                    return False
        return True

    def check_winner(self) -> Optional[Cell]:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(BOARD.rows):
            for c in range(BOARD.cols):
                piece = self.grid[r][c]
                if piece == Cell.EMPTY:
                    continue
                for dr, dc in directions:
                    count = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < BOARD.rows and 0 <= cc < BOARD.cols and self.grid[rr][cc] == piece:
                        count += 1
                        if count == 4:
                            return piece
                        rr += dr
                        cc += dc
        return None

    def is_full(self) -> bool:
        return all(not self.can_play(c) for c in range(BOARD.cols))


def board_from_top_down_strings(rows_top_down: List[List[str]]) -> Connect4Board:
    if len(rows_top_down) != BOARD.rows:
        raise ValueError(f"Expected {BOARD.rows} rows, got {len(rows_top_down)}")
    for row in rows_top_down:
        if len(row) != BOARD.cols:
            raise ValueError(f"Each row must have {BOARD.cols} columns")
    board = Connect4Board.empty()
    for input_r, row in enumerate(rows_top_down):
        internal_r = BOARD.rows - 1 - input_r
        for c, val in enumerate(row):
            board.grid[internal_r][c] = Cell(val)
    return board
