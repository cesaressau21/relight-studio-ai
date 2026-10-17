"""Undo/redo history built on document snapshots.

Rapid edits sharing a ``merge_key`` (a slider drag, a light drag) collapse into
one undo step when they arrive within ``merge_window`` seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .document import DocState


@dataclass
class _Entry:
    state: DocState
    label: str
    merge_key: str | None
    stamp: float


class History:
    def __init__(self, limit: int = 100, merge_window: float = 1.0) -> None:
        self._undo: list[_Entry] = []
        self._redo: list[_Entry] = []
        self.limit = limit
        self.merge_window = merge_window

    def reset(self, state: DocState) -> None:
        self._undo = [_Entry(state, "Open", None, time.monotonic())]
        self._redo.clear()

    @property
    def current(self) -> DocState | None:
        return self._undo[-1].state if self._undo else None

    def push(self, state: DocState, label: str, merge_key: str | None = None) -> bool:
        """Record a new state. Returns False when nothing changed."""
        now = time.monotonic()
        top = self._undo[-1] if self._undo else None
        if top is not None and top.state.same_params(state):
            return False
        mergeable = (
            top is not None
            and len(self._undo) > 1
            and merge_key is not None
            and top.merge_key == merge_key
            and now - top.stamp < self.merge_window
        )
        if mergeable:
            top.state, top.stamp = state, now  # type: ignore[union-attr]
        else:
            self._undo.append(_Entry(state, label, merge_key, now))
            if len(self._undo) > self.limit:
                del self._undo[1]  # keep the "Open" base state
        self._redo.clear()
        return True

    def can_undo(self) -> bool:
        return len(self._undo) > 1

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> DocState | None:
        if not self.can_undo():
            return None
        self._redo.append(self._undo.pop())
        return self._undo[-1].state

    def redo(self) -> DocState | None:
        if not self._redo:
            return None
        entry = self._redo.pop()
        entry.merge_key = None  # never merge into a redone step
        self._undo.append(entry)
        return entry.state

    def undo_label(self) -> str:
        return self._undo[-1].label if self.can_undo() else ""

    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""
