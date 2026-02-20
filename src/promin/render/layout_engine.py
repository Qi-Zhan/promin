from __future__ import annotations

import json
from typing import Any, Optional

from ..view import LayoutSpec, normalize_layout_spec
from .layout_registry import _require_layout
from .snapshot_view import _get_label, _get_node_color, _get_view, _is_container_snapshot, _resolve_snapshot
from .types import H_GAP, MAX_SCENE_WIDTH, V_GAP, LayoutContext, LayoutResult


class LayoutNode:
    """Intermediate layout element — a container with content."""

    __slots__ = (
        "node_id",
        "label",
        "focused",
        "type_name",
        "shape",
        "fill_color",
        "edge_styles",
        "x",
        "y",
        "children",
        "child_fields",
        "snapshot",
        "content_type",
        "content_root",
        "type_label",
        "layout_spec",
    )

    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.node_id: int | None = snapshot.get("_id")
        self.focused = snapshot.get("_focused", False)
        self.type_name = snapshot.get("_type", "")

        view = _get_view(snapshot)
        self.shape: str = view["shape"]
        self.label: str = _get_label(snapshot)
        self.fill_color: Optional[str] = _get_node_color(snapshot)
        raw_layout = view.get("layout")
        if raw_layout is None:
            raise ValueError(
                f"Missing layout for node type '{self.type_name}'. No default layout is allowed."
            )
        self.layout_spec: LayoutSpec = normalize_layout_spec(raw_layout)
        _require_layout(self.layout_spec.name)

        self.content_type: str = "text"
        self.content_root: Optional[LayoutNode] = None
        self.type_label: str = ""

        self.child_fields: list[str] = []
        self.children: list[Optional[LayoutNode]] = []
        self.edge_styles: list[dict] = []
        for edge_spec in view["edge_specs"]:
            field_name = edge_spec["field"]
            if field_name not in snapshot:
                continue
            child = snapshot[field_name]
            if isinstance(child, list):
                for item in child:
                    self.child_fields.append(field_name)
                    self.edge_styles.append(edge_spec)
                    if isinstance(item, dict) and "_type" in item:
                        self.children.append(LayoutNode(item))
                    else:
                        self.children.append(None)
            elif isinstance(child, dict) and "_type" in child:
                resolved = _resolve_snapshot(child)
                self.child_fields.append(field_name)
                self.edge_styles.append(edge_spec)
                if resolved is not None and isinstance(resolved, dict) and "_type" in resolved:
                    self.children.append(LayoutNode(resolved))
                else:
                    self.children.append(None)
            else:
                self.child_fields.append(field_name)
                self.edge_styles.append(edge_spec)
                self.children.append(None)


def layout_tree(snapshot: Any) -> Optional[LayoutNode]:
    snapshot = _resolve_snapshot(snapshot)
    if snapshot is None or not isinstance(snapshot, dict) or "_type" not in snapshot:
        return None

    if _is_container_snapshot(snapshot):
        view = _get_view(snapshot)
        inner = snapshot[view["content_field"]]
        inner_root = layout_tree(inner)
        if inner_root is None:
            return None
        container = LayoutNode.__new__(LayoutNode)
        container.snapshot = snapshot
        container.node_id = snapshot.get("_id")
        container.type_name = snapshot.get("_type", "")
        container.shape = view["shape"]
        container.fill_color = _get_node_color(snapshot)
        container.focused = snapshot.get("_focused", False)
        container.content_type = "subtree"
        container.content_root = inner_root
        container.label = ""
        container.type_label = container.type_name
        container.children = []
        container.child_fields = []
        container.edge_styles = []
        raw_layout = view.get("layout")
        if raw_layout is None:
            raise ValueError(
                f"Missing layout for node type '{container.type_name}'. No default layout is allowed."
            )
        container.layout_spec = normalize_layout_spec(raw_layout)
        _require_layout(container.layout_spec.name)
        container.x = 0.0
        container.y = 0.0
        return container

    root = LayoutNode(snapshot)
    _assign_positions(root, depth=0)
    _center_tree(root)
    _clamp_width(root)
    return root


def _assign_positions(node: LayoutNode, depth: int) -> None:
    node.x = 0.0
    node.y = -depth * V_GAP

    def _layout_for_edge(n: LayoutNode, spec: dict) -> LayoutSpec:
        raw = spec.get("layout")
        if raw is not None:
            return normalize_layout_spec(raw)
        if n.layout_spec is None:
            raise ValueError(
                f"Node '{n.type_name}' has no layout and edge '{spec.get('field')}' does not provide one."
            )
        return n.layout_spec

    def walk(n: LayoutNode, gap_x: float) -> None:
        real_pairs: list[tuple[LayoutNode, dict]] = [
            (c, spec) for c, spec in zip(n.children, n.edge_styles) if c is not None
        ]
        if not real_pairs:
            return

        specs = [_layout_for_edge(n, spec) for _, spec in real_pairs]
        key_set = {(s.name, json.dumps(s.params, sort_keys=True)) for s in specs}
        if len(key_set) != 1:
            raise ValueError(
                f"Node '{n.type_name}' has mixed child layouts. Use one layout per node level."
            )
        layout_spec = specs[0]
        layout_fn = _require_layout(layout_spec.name)
        ctx = LayoutContext(
            parent_id=n.node_id,
            children=[
                {
                    "node_id": child.node_id,
                    "field": spec.get("field"),
                    "direction": spec.get("direction", "auto"),
                    "index": i,
                }
                for i, (child, spec) in enumerate(real_pairs)
            ],
            params=dict(layout_spec.params),
            gap_x=gap_x,
            gap_y=V_GAP,
        )
        result = layout_fn(ctx)
        if not isinstance(result, LayoutResult):
            raise TypeError(
                f"Layout '{layout_spec.name}' must return LayoutResult, got {type(result).__name__}"
            )

        missing = [c.node_id for c, _ in real_pairs if c.node_id not in result.positions]
        if missing:
            raise ValueError(
                f"Layout '{layout_spec.name}' did not return coordinates for child ids: {missing}"
            )

        for child, _ in real_pairs:
            dx_dy = result.positions.get(child.node_id)
            if (
                not isinstance(dx_dy, tuple)
                or len(dx_dy) != 2
                or not isinstance(dx_dy[0], (int, float))
                or not isinstance(dx_dy[1], (int, float))
            ):
                raise ValueError(
                    f"Layout '{layout_spec.name}' returned invalid coordinate for child id {child.node_id}: {dx_dy!r}"
                )
            dx, dy = dx_dy
            child.x = n.x + float(dx)
            child.y = n.y + float(dy)
            next_gap_x = gap_x
            if layout_spec.name == "tree":
                next_gap_x = max(0.35, gap_x * 0.5)
            walk(child, next_gap_x)

    walk(node, 1.0)


def _center_tree(root: LayoutNode) -> None:
    xs = list(_collect_x(root))
    if not xs:
        return
    cx = (min(xs) + max(xs)) / 2.0
    _shift_x(root, -cx)


def _clamp_width(root: LayoutNode) -> None:
    xs = list(_collect_x(root))
    if not xs:
        return
    span = (max(xs) - min(xs)) * H_GAP
    if span > MAX_SCENE_WIDTH:
        factor = MAX_SCENE_WIDTH / span
        _scale_x(root, factor)


def _collect_x(node: LayoutNode):
    yield node.x
    for c in node.children:
        if c is not None:
            yield from _collect_x(c)


def _shift_x(node: LayoutNode, dx: float):
    node.x += dx
    for c in node.children:
        if c is not None:
            _shift_x(c, dx)


def _scale_x(node: LayoutNode, factor: float):
    node.x *= factor
    for c in node.children:
        if c is not None:
            _scale_x(c, factor)


def _flatten_nodes(node: Optional[LayoutNode]) -> list[LayoutNode]:
    if node is None:
        return []
    result = [node]
    for c in node.children:
        result.extend(_flatten_nodes(c))
    return result
