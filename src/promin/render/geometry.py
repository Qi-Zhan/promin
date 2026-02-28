from __future__ import annotations

import numpy as np

from .layout_engine import LayoutNode, _flatten_nodes
from .types import BOX_HEIGHT, BOX_WIDTH, H_GAP, NODE_RADIUS


def _node_pos(node: LayoutNode, origin: np.ndarray) -> np.ndarray:
    return origin + np.array([node.x * H_GAP, node.y, 0.0])


def _boundary_offset(
    shape: str | None,
    direction_unit: np.ndarray,
    *,
    width: float | None = None,
    height: float | None = None,
) -> np.ndarray:
    dx = float(direction_unit[0])
    dy = float(direction_unit[1])

    if shape is None:
        return np.array([0.0, 0.0, 0.0])

    if shape == "box":
        hw = float(width) / 2.0 if width is not None else BOX_WIDTH / 2.0
        hh = float(height) / 2.0 if height is not None else BOX_HEIGHT / 2.0
        tx = float("inf") if abs(dx) < 1e-9 else hw / abs(dx)
        ty = float("inf") if abs(dy) < 1e-9 else hh / abs(dy)
        t = min(tx, ty)
        return np.array([dx * t, dy * t, 0.0])

    if shape == "diamond":
        default = NODE_RADIUS * 2.4
        r = max(float(width) if width is not None else default, float(height) if height is not None else default) / 2.0
        denom = abs(dx) + abs(dy)
        if denom < 1e-9:
            return np.array([0.0, 0.0, 0.0])
        t = r / denom
        return np.array([dx * t, dy * t, 0.0])

    default = NODE_RADIUS * 2.0
    r = max(float(width) if width is not None else default, float(height) if height is not None else default) / 2.0
    return np.array([dx * r, dy * r, 0.0])


def _compute_bounding_box(root: LayoutNode, origin: np.ndarray) -> tuple[float, float, float, float]:
    all_nodes = _flatten_nodes(root)
    positions = [_node_pos(n, origin) for n in all_nodes]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return min(xs), max(xs), min(ys), max(ys)
