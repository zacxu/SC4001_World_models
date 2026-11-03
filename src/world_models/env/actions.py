"""Fixed categorical actions shared by both players and B/D variants.

No function in this module consults simulator state or generates action masks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from world_models.contracts import ACTION_BUTTONS, validate_action


BUTTON_NAMES: tuple[str, ...] = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
)


def encode_action(action: int, button_names: Sequence[str] = BUTTON_NAMES) -> list[int]:
    """Return the eight engine buttons for one of the 12 executable actions.

    A different order is permitted only as an exact permutation of the declared
    buttons. The reset START symbol is history metadata, never an engine action.
    """
    index = validate_action(action)
    if isinstance(button_names, (str, bytes)) or not isinstance(button_names, Sequence):
        raise ValueError("button_names must be an ordered sequence of declared names")
    names = tuple(button_names)
    if len(names) != len(BUTTON_NAMES) or any(type(name) is not str for name in names):
        raise ValueError("button_names must contain the eight declared button names")
    if set(names) != set(BUTTON_NAMES):
        raise ValueError("button_names must be an exact permutation without duplicates")
    pressed = frozenset(ACTION_BUTTONS[index])
    return [int(name in pressed) for name in names]


def button_enums() -> list[Any]:
    """Resolve ViZDoom enums lazily, keeping non-simulator imports lightweight."""
    import vizdoom

    return [getattr(vizdoom.Button, name) for name in BUTTON_NAMES]

