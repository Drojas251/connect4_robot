from __future__ import annotations

from typing import Optional

from .board import Connect4Board
from .messages import MoveDecision


class WeightedRandomPolicy:
    def __init__(self):
        self.last_col: Optional[int] = None

    def choose_move(self, board: Connect4Board) -> MoveDecision:
        import random
        legal = board.legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available.")
        candidates = [c for c in legal if c != self.last_col] or legal
        weights_map = {3: 5, 2: 4, 4: 4, 1: 3, 5: 3, 0: 2, 6: 2}
        weights = [weights_map[c] for c in candidates]
        col = random.choices(candidates, weights=weights, k=1)[0]
        self.last_col = col
        return MoveDecision(column=col, reason="weighted-random-demo-policy")
