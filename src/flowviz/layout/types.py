from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclass(frozen=True)
class Anchor:
    id: int | str
    pos: Position
    meta: dict[str, Any] = field(default_factory=dict)

    def with_pos(self, pos: Position) -> "Anchor":
        return Anchor(id=self.id, pos=pos, meta=dict(self.meta))


@dataclass
class LinksLayoutContext:
    parent_id: int | None
    gap_x: float
    gap_y: float
    level: int = 0
    params: dict[str, Any] = field(default_factory=dict)
