"""
Connect4AI — plugin interface for robot move strategies.

To create a custom AI
---------------------
1. Subclass Connect4AI and set class-level ``name`` and ``description``.
2. Implement ``choose_move()``.
3. Register it with the ``@register_ai`` decorator (imported from ai_registry).

Minimal example::

    from connect4_robot.game_engine.ai_base import Connect4AI
    from connect4_robot.game_engine.ai_registry import register_ai
    from connect4_robot.game_engine.board import Cell
    from connect4_robot.game_engine.messages import MoveDecision
    import random

    @register_ai
    class MyAI(Connect4AI):
        name        = "my_ai"
        description = "Does something clever."

        def choose_move(self, board, robot_piece=Cell.ROBOT):
            col = random.choice(board.legal_moves())
            return MoveDecision(column=col, reason="my_ai/random")

Once registered the AI appears in ``GET /ai`` and can be selected at runtime
via ``POST /ai`` without restarting services.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .board import Cell, Connect4Board
from .messages import MoveDecision


class Connect4AI(ABC):
    """Abstract base class every robot AI must implement."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @abstractmethod
    def choose_move(
        self,
        board: Connect4Board,
        robot_piece: Cell = Cell.ROBOT,
    ) -> MoveDecision:
        """Return the column the robot should play next."""
        ...

    def on_game_reset(self) -> None:
        """Called at the start of each new game. Override to reset per-game state."""
