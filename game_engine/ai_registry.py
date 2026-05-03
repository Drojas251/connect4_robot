"""
AI registry — central catalogue of available Connect4AI implementations.

Usage
-----
Register (usually done at module level with a decorator)::

    from connect4_robot.game_engine.ai_registry import register_ai

    @register_ai
    class MyAI(Connect4AI):
        name = "my_ai"
        ...

Instantiate by name::

    from connect4_robot.game_engine.ai_registry import build_ai
    ai = build_ai("minimax", difficulty="hard")

List what's available::

    from connect4_robot.game_engine.ai_registry import list_ais
    print(list_ais())
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from .ai_base import Connect4AI

_REGISTRY: Dict[str, "Type[Connect4AI]"] = {}
_builtins_loaded = False


def register_ai(cls: "Type[Connect4AI]") -> "Type[Connect4AI]":
    """Class decorator — adds *cls* to the registry under ``cls.name``."""
    if not cls.name:
        raise ValueError(
            f"{cls.__name__} must set a non-empty class-level `name` attribute."
        )
    _REGISTRY[cls.name] = cls
    return cls


def get_ai_class(name: str) -> "Type[Connect4AI]":
    """Return the class registered under *name*, loading builtins first."""
    _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown AI '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def build_ai(name: str, **params) -> "Connect4AI":
    """Instantiate a registered AI, forwarding *params* to its constructor."""
    return get_ai_class(name)(**params)


def list_ais() -> List[dict]:
    """Return a list of ``{name, description}`` dicts for all registered AIs."""
    _load_builtins()
    return [
        {"name": cls.name, "description": cls.description}
        for cls in _REGISTRY.values()
    ]


def _load_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    from . import ai_minimax  # noqa: F401  triggers @register_ai
    from . import policy       # noqa: F401  triggers @register_ai
