from __future__ import annotations

import importlib
import inspect
import textwrap
from typing import Any, Callable

import numpy as np

from .types import LayoutContext, LayoutResult

LayoutFn = Callable[[LayoutContext], LayoutResult]
_layout_registry: dict[str, LayoutFn] = {}
_builtin_layout_names: set[str] = set()


def register_layout(name: str, fn: LayoutFn) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError("layout name must be a non-empty string")
    if not callable(fn):
        raise TypeError("layout function must be callable")
    _layout_registry[name] = fn


def _layout_row(ctx: LayoutContext) -> LayoutResult:
    items = [c for c in ctx.children if c.get("node_id") is not None]
    n = len(items)
    if n == 0:
        return LayoutResult({})
    wrap = bool(ctx.params.get("wrap", False))
    columns = ctx.params.get("columns")
    if columns is None:
        columns = n
    if not isinstance(columns, int) or columns <= 0:
        raise ValueError("row layout requires params.columns to be a positive int")
    cols = min(columns, n) if wrap else n
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


def _layout_column(ctx: LayoutContext) -> LayoutResult:
    items = [c for c in ctx.children if c.get("node_id") is not None]
    out: dict[int, tuple[float, float]] = {}
    for idx, child in enumerate(items):
        out[child["node_id"]] = (0.0, -(idx + 1) * ctx.gap_y)
    return LayoutResult(out)


def _layout_grid(ctx: LayoutContext) -> LayoutResult:
    items = [c for c in ctx.children if c.get("node_id") is not None]
    n = len(items)
    if n == 0:
        return LayoutResult({})
    columns = ctx.params.get("columns", 3)
    if not isinstance(columns, int) or columns <= 0:
        raise ValueError("grid layout requires params.columns to be a positive int")
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


def _layout_radial(ctx: LayoutContext) -> LayoutResult:
    items = [c for c in ctx.children if c.get("node_id") is not None]
    n = len(items)
    if n == 0:
        return LayoutResult({})
    radius = float(ctx.params.get("radius", max(ctx.gap_x, ctx.gap_y)))
    start_angle = float(ctx.params.get("start_angle", -np.pi / 2))
    step = (2 * np.pi) / n
    out: dict[int, tuple[float, float]] = {}
    for i, child in enumerate(items):
        theta = start_angle + i * step
        out[child["node_id"]] = (radius * np.cos(theta), radius * np.sin(theta) - ctx.gap_y)
    return LayoutResult(out)


def _layout_tree(ctx: LayoutContext) -> LayoutResult:
    items = [c for c in ctx.children if c.get("node_id") is not None]
    if not items:
        return LayoutResult({})

    left: list[dict[str, Any]] = []
    right: list[dict[str, Any]] = []
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


def _require_layout(name: str) -> LayoutFn:
    fn = _layout_registry.get(name)
    if fn is None:
        available = ", ".join(sorted(_layout_registry.keys()))
        raise ValueError(f"Unknown layout '{name}'. Available layouts: {available}")
    return fn


def _register_builtin_layouts() -> None:
    register_layout("tree", _layout_tree)
    register_layout("row", _layout_row)
    register_layout("column", _layout_column)
    register_layout("grid", _layout_grid)
    register_layout("radial", _layout_radial)
    _builtin_layout_names.update({"tree", "row", "column", "grid", "radial"})


def _custom_layout_bootstrap_code() -> str:
    """Generate Python code to re-register custom layouts in the render subprocess."""
    lines: list[str] = []
    for name, fn in _layout_registry.items():
        if name in _builtin_layout_names:
            continue

        qualname = getattr(fn, "__qualname__", "")
        module = getattr(fn, "__module__", "")
        if module and module != "__main__" and "<locals>" not in qualname:
            lines.append(f"_mod = importlib.import_module({module!r})")
            lines.append("_fn = _mod")
            for part in qualname.split("."):
                lines.append(f"_fn = getattr(_fn, {part!r})")
            lines.append(f"register_layout({name!r}, _fn)")
            continue

        try:
            source = textwrap.dedent(inspect.getsource(fn)).rstrip()
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot serialize custom layout '{name}'. "
                "Define it as a module-level function."
            ) from exc
        lines.append(source)
        lines.append(f"register_layout({name!r}, {fn.__name__})")

    return "\n".join(lines)


_register_builtin_layouts()
