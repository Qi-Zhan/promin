from __future__ import annotations

from math import pi
from typing import Callable

import numpy as np

from .types import LayoutContext, LayoutResult

LayoutCallable = Callable[[LayoutContext], LayoutResult]


def _mark_layout(fn: LayoutCallable, kind: str) -> LayoutCallable:
    setattr(fn, "_promin_layout_kind", kind)
    return fn


def _tree_layout(ctx: LayoutContext) -> LayoutResult:
    items = [c for c in ctx.children if c.get("node_id") is not None]
    if not items:
        return LayoutResult({})

    left: list[dict] = []
    right: list[dict] = []
    for child in items:
        direction = child.get("direction", "auto")
        if direction == "left":
            left.append(child)
        elif direction == "right":
            right.append(child)
        else:
            left.append(child)

    if all(c.get("direction", "auto") in ("auto", "down", "up") for c in items):
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

    out: dict[int, tuple[float, float]] = {}
    for i, child in enumerate(reversed(left), start=1):
        out[child["node_id"]] = (-i * ctx.gap_x, -ctx.gap_y)
    for i, child in enumerate(right, start=1):
        out[child["node_id"]] = (i * ctx.gap_x, -ctx.gap_y)
    return LayoutResult(out)


TreeLayout: LayoutCallable = _mark_layout(_tree_layout, "tree")


def RowLayout(wrap: bool = False, columns: int | None = None) -> LayoutCallable:
    def _layout(ctx: LayoutContext) -> LayoutResult:
        items = [c for c in ctx.children if c.get("node_id") is not None]
        n = len(items)
        if n == 0:
            return LayoutResult({})
        cols = columns if columns is not None else n
        if not isinstance(cols, int) or cols <= 0:
            raise ValueError("RowLayout(columns=...) requires a positive int when provided")
        cols = min(cols, n) if wrap else n
        out: dict[int, tuple[float, float]] = {}
        rows = (n + cols - 1) // cols
        for idx, child in enumerate(items):
            row = idx // cols
            col = idx % cols
            row_count = cols if row < rows - 1 else (n - row * cols)
            x = (col - (row_count - 1) / 2.0) * ctx.gap_x
            y = -(row + 1) * ctx.gap_y
            out[child["node_id"]] = (x, y)
        return LayoutResult(out)

    return _mark_layout(_layout, "row")


def ColumnLayout() -> LayoutCallable:
    def _layout(ctx: LayoutContext) -> LayoutResult:
        items = [c for c in ctx.children if c.get("node_id") is not None]
        out: dict[int, tuple[float, float]] = {}
        for idx, child in enumerate(items):
            out[child["node_id"]] = (0.0, -(idx + 1) * ctx.gap_y)
        return LayoutResult(out)

    return _mark_layout(_layout, "column")


def GridLayout(columns: int = 3) -> LayoutCallable:
    def _layout(ctx: LayoutContext) -> LayoutResult:
        items = [c for c in ctx.children if c.get("node_id") is not None]
        n = len(items)
        if n == 0:
            return LayoutResult({})
        if not isinstance(columns, int) or columns <= 0:
            raise ValueError("GridLayout(columns=...) requires a positive int")
        out: dict[int, tuple[float, float]] = {}
        rows = (n + columns - 1) // columns
        for idx, child in enumerate(items):
            row = idx // columns
            col = idx % columns
            row_count = columns if row < rows - 1 else (n - row * columns)
            x = (col - (row_count - 1) / 2.0) * ctx.gap_x
            y = -(row + 1) * ctx.gap_y
            out[child["node_id"]] = (x, y)
        return LayoutResult(out)

    return _mark_layout(_layout, "grid")


def RadialLayout(radius: float | None = None, start_angle: float = -pi / 2) -> LayoutCallable:
    def _layout(ctx: LayoutContext) -> LayoutResult:
        items = [c for c in ctx.children if c.get("node_id") is not None]
        n = len(items)
        if n == 0:
            return LayoutResult({})
        rad = float(radius) if radius is not None else float(max(ctx.gap_x, ctx.gap_y))
        step = (2 * np.pi) / n
        out: dict[int, tuple[float, float]] = {}
        for i, child in enumerate(items):
            theta = float(start_angle) + i * step
            out[child["node_id"]] = (rad * np.cos(theta), rad * np.sin(theta) - ctx.gap_y)
        return LayoutResult(out)

    return _mark_layout(_layout, "radial")
