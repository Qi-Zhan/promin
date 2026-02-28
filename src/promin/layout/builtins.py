from __future__ import annotations

from math import pi
from typing import Callable

import numpy as np

from .types import Anchor, Position

LinksLayoutFn = Callable[[list[Anchor], Anchor, object], list[Anchor]]


def _mark_layout(fn: Callable, kind: str) -> Callable:
    setattr(fn, "_promin_layout_kind", kind)
    return fn


def row(wrap: bool = False, columns: int | None = None) -> LinksLayoutFn:
    def _layout(targets: list[Anchor], origin: Anchor, ctx: object) -> list[Anchor]:
        n = len(targets)
        if n == 0:
            return []
        gap_x = float(getattr(ctx, "gap_x", 1.0))
        gap_y = float(getattr(ctx, "gap_y", 1.0))
        cols = columns if columns is not None else n
        if not isinstance(cols, int) or cols <= 0:
            raise ValueError("row(columns=...) requires a positive int when provided")
        cols = min(cols, n) if wrap else n

        out: list[Anchor] = []
        rows = (n + cols - 1) // cols
        for idx, t in enumerate(targets):
            r = idx // cols
            c = idx % cols
            row_count = cols if r < rows - 1 else (n - r * cols)
            x = origin.pos.x + (c - (row_count - 1) / 2.0) * gap_x
            y = origin.pos.y - (r + 1) * gap_y
            out.append(t.with_pos(Position(x=x, y=y)))
        return out

    return _mark_layout(_layout, "links_row")


def grid(columns: int) -> LinksLayoutFn:
    if not isinstance(columns, int) or columns <= 0:
        raise ValueError("grid(columns=...) requires a positive int")
    return row(wrap=True, columns=columns)


def column() -> LinksLayoutFn:
    def _layout(targets: list[Anchor], origin: Anchor, ctx: object) -> list[Anchor]:
        gap_y = float(getattr(ctx, "gap_y", 1.0))
        return [
            t.with_pos(Position(x=origin.pos.x, y=origin.pos.y - (i + 1) * gap_y))
            for i, t in enumerate(targets)
        ]

    return _mark_layout(_layout, "links_column")


def _tree_layout(targets: list[Anchor], origin: Anchor, ctx: object) -> list[Anchor]:
    if not targets:
        return []
    gap_x = float(getattr(ctx, "gap_x", 1.0))
    gap_y = float(getattr(ctx, "gap_y", 1.0))

    left: list[Anchor] = []
    right: list[Anchor] = []
    for t in targets:
        hint = t.meta.get("hint", "")
        if hint == "left":
            left.append(t)
        elif hint == "right":
            right.append(t)
        else:
            left.append(t)

    if all(t.meta.get("hint", "") in ("", "auto") for t in targets):
        mid = len(targets) // 2
        left = targets[:mid]
        right = targets[mid:]

    out: list[Anchor] = []
    for i, t in enumerate(reversed(left), start=1):
        out.append(t.with_pos(Position(origin.pos.x - i * gap_x, origin.pos.y - gap_y)))
    for i, t in enumerate(right, start=1):
        out.append(t.with_pos(Position(origin.pos.x + i * gap_x, origin.pos.y - gap_y)))

    mapping = {a.id: a for a in out}
    return [mapping[t.id] for t in targets]


tree: LinksLayoutFn = _mark_layout(_tree_layout, "links_tree")


def radial(radius: float | None = None, start_angle: float = -pi / 2) -> LinksLayoutFn:
    def _layout(targets: list[Anchor], origin: Anchor, ctx: object) -> list[Anchor]:
        n = len(targets)
        if n == 0:
            return []
        gap_x = float(getattr(ctx, "gap_x", 1.0))
        gap_y = float(getattr(ctx, "gap_y", 1.0))
        rad = float(radius) if radius is not None else float(max(gap_x, gap_y))
        step = (2 * np.pi) / n
        out: list[Anchor] = []
        for i, t in enumerate(targets):
            th = float(start_angle) + i * step
            out.append(
                t.with_pos(
                    Position(
                        x=origin.pos.x + rad * float(np.cos(th)),
                        y=origin.pos.y + rad * float(np.sin(th)) - gap_y,
                    )
                )
            )
        return out

    return _mark_layout(_layout, "links_radial")
