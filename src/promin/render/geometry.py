from __future__ import annotations

import numpy as np

from .layout_engine import LayoutNode, _flatten_nodes
from .types import BOX_HEIGHT, BOX_WIDTH, EDGE_COLOR, EDGE_STROKE, H_GAP, NODE_RADIUS


def _node_pos(node: LayoutNode, origin: np.ndarray) -> np.ndarray:
    return origin + np.array([node.x * H_GAP, node.y, 0.0])


def _node_radius(shape: str) -> float:
    if shape == "box":
        return BOX_WIDTH / 2
    if shape == "diamond":
        return NODE_RADIUS * 1.2
    return NODE_RADIUS


def _boundary_offset(shape: str, direction_unit: np.ndarray) -> np.ndarray:
    dx = float(direction_unit[0])
    dy = float(direction_unit[1])

    if shape == "box":
        hw = BOX_WIDTH / 2.0
        hh = BOX_HEIGHT / 2.0
        tx = float("inf") if abs(dx) < 1e-9 else hw / abs(dx)
        ty = float("inf") if abs(dy) < 1e-9 else hh / abs(dy)
        t = min(tx, ty)
        return np.array([dx * t, dy * t, 0.0])

    if shape == "diamond":
        r = NODE_RADIUS * 1.2
        denom = abs(dx) + abs(dy)
        if denom < 1e-9:
            return np.array([0.0, 0.0, 0.0])
        t = r / denom
        return np.array([dx * t, dy * t, 0.0])

    r = _node_radius(shape)
    return np.array([dx * r, dy * r, 0.0])


def _compute_bounding_box(root: LayoutNode, origin: np.ndarray) -> tuple[float, float, float, float]:
    all_nodes = _flatten_nodes(root)
    positions = [_node_pos(n, origin) for n in all_nodes]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return min(xs), max(xs), min(ys), max(ys)
