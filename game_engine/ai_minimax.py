"""
MinimaxAI — alpha-beta minimax with three difficulty levels.

Difficulty  Depth  Random-move rate  Character
----------  -----  ----------------  --------------------------
easy          2       25 %           Makes blunders, fun for beginners
medium        5        0 %           Sees 2-3 moves ahead, competitive
hard         10        0 %           Near-perfect on a 5×5 board
"""
from __future__ import annotations

import random
from typing import ClassVar, List, Tuple

from connect4_robot.config import BOARD

from .ai_base import Connect4AI
from .ai_registry import register_ai
from .board import Cell, Connect4Board
from .messages import MoveDecision

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF = float("inf")

_DEPTH: dict[str, int] = {"easy": 2, "medium": 5, "hard": 10}
_RANDOM_RATE: dict[str, float] = {"easy": 0.25, "medium": 0.0, "hard": 0.0}

# Points for an unblocked window of N pieces
_WINDOW_SCORE: dict[int, int] = {4: 1_000_000, 3: 50, 2: 5, 1: 1}


# ---------------------------------------------------------------------------
# Pure helpers (module-level so they're easy to unit-test)
# ---------------------------------------------------------------------------

def _move_order(legal: List[int]) -> List[int]:
    """Sort columns closest-to-center first — improves alpha-beta cutoffs."""
    center = BOARD.cols // 2
    return sorted(legal, key=lambda c: abs(c - center))


def _score_window(window: List[Cell], piece: Cell, opp: Cell) -> int:
    if opp in window:
        return 0
    return _WINDOW_SCORE.get(window.count(piece), 0)


def _evaluate(board: Connect4Board, robot: Cell, human: Cell) -> int:
    """Heuristic board score from robot's perspective."""
    rows, cols = BOARD.rows, BOARD.cols
    score = 0

    # Center column is the most valuable real estate
    center = cols // 2
    for r in range(rows):
        if board.grid[r][center] == robot:
            score += 3
        elif board.grid[r][center] == human:
            score -= 3

    # Score every length-4 window in all four directions
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                window: List[Cell] = []
                for i in range(4):
                    rr, cc = r + dr * i, c + dc * i
                    if 0 <= rr < rows and 0 <= cc < cols:
                        window.append(board.grid[rr][cc])
                if len(window) == 4:
                    score += _score_window(window, robot, human)
                    score -= _score_window(window, human, robot)

    return score


def _minimax(
    board: Connect4Board,
    depth: int,
    is_max: bool,
    alpha: float,
    beta: float,
    robot: Cell,
    human: Cell,
) -> float:
    winner = board.check_winner()
    if winner == robot:
        return _INF
    if winner == human:
        return -_INF
    if board.is_full() or depth == 0:
        return _evaluate(board, robot, human)

    piece = robot if is_max else human
    best = -_INF if is_max else _INF

    for col in _move_order(board.legal_moves()):
        child = board.copy()
        child.apply_move(col, piece)
        val = _minimax(child, depth - 1, not is_max, alpha, beta, robot, human)

        if is_max:
            best = max(best, val)
            alpha = max(alpha, best)
        else:
            best = min(best, val)
            beta = min(beta, best)

        if alpha >= beta:
            break  # prune

    return best


# ---------------------------------------------------------------------------
# Registered AI
# ---------------------------------------------------------------------------

@register_ai
class MinimaxAI(Connect4AI):
    """Alpha-beta minimax with configurable difficulty."""

    name: ClassVar[str] = "minimax"
    description: ClassVar[str] = (
        "Alpha-beta minimax — easy (depth 2 + 25% random), "
        "medium (depth 5), hard (depth 10 / near-perfect)"
    )

    DIFFICULTIES: ClassVar[List[str]] = ["easy", "medium", "hard"]

    def __init__(self, difficulty: str = "medium") -> None:
        if difficulty not in _DEPTH:
            raise ValueError(
                f"difficulty must be one of {self.DIFFICULTIES}, got '{difficulty}'"
            )
        self.difficulty = difficulty
        self._depth = _DEPTH[difficulty]
        self._random_rate = _RANDOM_RATE[difficulty]

    def choose_move(
        self,
        board: Connect4Board,
        robot_piece: Cell = Cell.ROBOT,
    ) -> MoveDecision:
        legal = board.legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available.")

        human_piece = Cell.HUMAN if robot_piece == Cell.ROBOT else Cell.ROBOT

        # Easy-mode blunder: occasionally play randomly
        if self._random_rate > 0 and random.random() < self._random_rate:
            col = random.choice(legal)
            return MoveDecision(column=col, reason=f"minimax/{self.difficulty}/random")

        ordered = _move_order(legal)
        best_col = ordered[0]
        best_score = -_INF

        for col in ordered:
            child = board.copy()
            child.apply_move(col, robot_piece)
            score = _minimax(
                child, self._depth - 1, False, -_INF, _INF, robot_piece, human_piece
            )
            if score > best_score:
                best_score = score
                best_col = col

        return MoveDecision(
            column=best_col,
            reason=f"minimax/{self.difficulty}/score={int(best_score) if best_score != _INF else 'WIN'}",
        )

    def on_game_reset(self) -> None:
        pass  # stateless between games
