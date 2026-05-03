from __future__ import annotations

import random
from typing import ClassVar, Optional

from .ai_base import Connect4AI
from .ai_registry import register_ai
from .board import Cell, Connect4Board
from .messages import MoveDecision

# Column weights for a 5-column board — centre is best
_WEIGHTS = {0: 2, 1: 3, 2: 5, 3: 3, 4: 2}


@register_ai
class WeightedRandomPolicy(Connect4AI):
    """Weighted-random: prefers centre columns, no lookahead."""

    name: ClassVar[str] = "random"
    description: ClassVar[str] = (
        "Weighted random — prefers centre columns, no lookahead. "
        "Good for testing the physical robot without strategic play."
    )

    def __init__(self) -> None:
        self._last_col: Optional[int] = None

    def choose_move(
        self,
        board: Connect4Board,
        robot_piece: Cell = Cell.ROBOT,
    ) -> MoveDecision:
        legal = board.legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available.")
        candidates = [c for c in legal if c != self._last_col] or legal
        weights = [_WEIGHTS.get(c, 1) for c in candidates]
        col = random.choices(candidates, weights=weights, k=1)[0]
        self._last_col = col
        return MoveDecision(column=col, reason="weighted-random")

    def on_game_reset(self) -> None:
        self._last_col = None
