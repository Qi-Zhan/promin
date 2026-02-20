"""
promin.render — Manim-based visualization of state sequences.

Two entry points:

* ``render_tree_text(snapshot)``  — quick text dump (debug / fallback)
* ``render_states(states, path)`` — produce a manim video file

The renderer is **data-driven**: each snapshot node carries a ``_view`` dict
that specifies the shape, label field, and edge fields.  No hardcoded
assumptions about field names (like ``"key"`` or ``"left"``).
"""

from __future__ import annotations

import logging
import json
import subprocess
import sys
import tempfile
import textwrap
import inspect
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .view import LayoutSpec, normalize_layout_spec, View

from manim import (
    BLUE_C,
    DOWN,
    GREY_A,
    GREY_B,
    RIGHT,
    UP,
    WHITE,
    BLACK as MANIM_BLACK,
    YELLOW_C,
    Arrow,
    Circle,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Polygon,
    ReplacementTransform,
    Scene,
    Text,
    VGroup,
    VMobject,
    ManimColor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RenderConfig — user-facing rendering options
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class RenderConfig:
    """Rendering options passed to ``StateMachine.render()``.

    All fields have sensible defaults so you only need to specify
    what you want to change.

    Attributes
    ----------
    background_color : str
        Scene background color (hex string, e.g. ``"#FFFFFF"`` for white).
        Default is manim's default (``"#000000"``, black).
    node_color : str
        Default node fill color (hex).  Overridden by per-node ``color_field``.
    edge_color : str
        Color for edges / arrows (hex).
    title_color : str
        Color for the title text (hex).
    text_color : str
        Color for overlay text (location, counter).  ``"auto"`` picks
        a contrasting color based on ``background_color``.
    quality : str
        Manim quality flag: ``"l"`` (low), ``"m"`` (medium),
        ``"h"`` (high), ``"k"`` (4K).  Default ``"l"``.
    """

    background_color: str = ""  # empty → manim default (black)
    node_color: str = ""       # empty → use NORMAL_FILL constant
    edge_color: str = ""       # empty → GREY_B
    title_color: str = ""      # empty → YELLOW_C
    text_color: str = "auto"   # auto → contrast against background
    quality: str = "l"         # l | m | h | k


@dataclass
class LayoutContext:
    """Context passed to layout plugins."""

    parent_id: int | None
    children: list[dict[str, Any]]
    params: dict[str, Any]
    gap_x: float
    gap_y: float


@dataclass
class LayoutResult:
    """Relative child coordinates keyed by child node id."""

    positions: dict[int, tuple[float, float]] = dc_field(default_factory=dict)


LayoutFn = Callable[[LayoutContext], LayoutResult]
_layout_registry: dict[str, LayoutFn] = {}
_builtin_layout_names: set[str] = set()


def register_layout(name: str, fn: LayoutFn) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError("layout name must be a non-empty string")
    if not callable(fn):
        raise TypeError("layout function must be callable")
    _layout_registry[name] = fn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_RADIUS = 0.30
BOX_WIDTH = 0.60
BOX_HEIGHT = 0.50
H_GAP = 1.3  # horizontal spacing multiplier
V_GAP = 1.1  # vertical spacing between levels
ANIM_DURATION = 0.45  # base animation duration
MAX_SCENE_WIDTH = 12.0  # clamp tree width to fit the frame

FOCUS_COLOR = YELLOW_C
FOCUS_STROKE = 3.5
NORMAL_FILL = BLUE_C
NORMAL_FILL_OPACITY = 0.22
FOCUS_FILL_OPACITY = 0.45
EDGE_COLOR = GREY_B
EDGE_STROKE = 1.8


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
    radius = float(ctx.params.get("radius", 1.8))
    out: dict[int, tuple[float, float]] = {}
    for idx, child in enumerate(items):
        theta = 2 * np.pi * idx / n
        out[child["node_id"]] = (radius * np.cos(theta), -abs(radius * np.sin(theta)))
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


_register_builtin_layouts()


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


# ---------------------------------------------------------------------------
# _NodeRenderInfo — unified visual element descriptor
# ---------------------------------------------------------------------------


@dataclass
class _NodeRenderInfo:
    """Unified render info for any visual element.

    Both leaf nodes (a circle with text "7") and container borders (a box
    around a subtree) are described by the same structure.  The renderer
    creates and diffs mobjects from these uniformly.

    Attributes
    ----------
    width, height : float | None
        When ``None`` the default size for the shape is used (leaf node).
        When set, the shape is explicitly sized (container border).
    type_label : str
        Corner label for containers (e.g. "RedBlackTree").
    """

    node_id: int
    pos: np.ndarray
    shape: str
    fill_color: Optional[str] = None
    focused: bool = False
    text: str = ""
    width: Optional[float] = None
    height: Optional[float] = None
    type_label: str = ""
    z_index: int = 0


# ---------------------------------------------------------------------------
# Snapshot helpers — read _view metadata
# ---------------------------------------------------------------------------


def _get_view(snapshot: dict) -> dict:
    """Return the ``_view`` dict from a snapshot node, with defaults."""
    view = snapshot.get("_view", {})
    return {
        "shape": view.get("shape", "circle"),
        "label": view.get("label", ""),
        "edges": [e["field"] for e in view.get("edges", [])],
        "edge_specs": view.get("edges", []),
        "data": view.get("data", []),
        "color_field": view.get("color_field", ""),
        "color_map": view.get("color_map", {}),
        "layout": view.get("layout"),
        "content_field": view.get("content_field", ""),
    }


def _get_label(snapshot: dict) -> str:
    """Return the display label for a snapshot node."""
    view = _get_view(snapshot)
    label_field = view["label"]
    if label_field and label_field in snapshot:
        return _format_label_value(snapshot[label_field])
    return str(snapshot.get("_type", "?"))


def _get_node_color(snapshot: dict) -> Optional[str]:
    """Return the resolved fill color string for a snapshot node, or None."""
    view = _get_view(snapshot)
    color_field = view["color_field"]
    if not color_field:
        return None
    field_value = snapshot.get(color_field)
    if field_value is None:
        return None
    color_map = view["color_map"]
    if color_map:
        return color_map.get(str(field_value))
    # If no map, try using the raw value as a color string
    return str(field_value)


def _contrast_text_color(fill_hex: str) -> str:
    """Return WHITE or BLACK text color for readability on *fill_hex*.

    Uses the W3C relative luminance formula.
    """
    hex_str = fill_hex.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    try:
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    except (ValueError, IndexError):
        return WHITE  # fallback
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return MANIM_BLACK if luminance > 0.5 else WHITE


def _format_label_value(val: Any) -> str:
    """Format a snapshot label value for display.

    Registered-class snapshot dicts (those with ``_type``) are handled
    specially because they are serialized dicts rather than live objects.
    Everything else is delegated to :meth:`View.format_value`, which
    dispatches through the View system.
    """
    if isinstance(val, dict) and "_type" in val:
        # Serialized registered object — drill into its own label field
        inner_view = _get_view(val)
        inner_label = inner_view["label"]
        if inner_label and inner_label in val:
            return _format_label_value(val[inner_label])
        return str(val.get("_type", "?"))
    # Delegate primitives and containers to the View dispatch system
    return View.format_value(val)


# ---------------------------------------------------------------------------
# Transparent wrapper resolution
# ---------------------------------------------------------------------------


def _resolve_snapshot(snapshot: Any) -> Any:
    """Unwrap transparent wrapper nodes (``shape=None``) to their content subtree.

    A registered class with ``shape=None`` acts as a transparent container
    whose ``label`` field is itself a registered class.  This function
    recursively follows the label field until it finds a concrete node
    (one with a real shape) or ``None``.
    """
    if not isinstance(snapshot, dict) or "_type" not in snapshot:
        return snapshot
    view = _get_view(snapshot)
    if view["shape"] is not None:
        return snapshot
    label_field = view["label"]
    if label_field and label_field in snapshot:
        inner = snapshot[label_field]
        return _resolve_snapshot(inner)
    return None


def _is_container_snapshot(snapshot: dict) -> bool:
    """Container is explicit via content_field, never inferred from label."""
    view = _get_view(snapshot)
    if view["shape"] is None:
        return False  # transparent wrapper, not a container
    content_field = view["content_field"]
    if not content_field or content_field not in snapshot:
        return False
    inner = snapshot[content_field]
    if not (isinstance(inner, dict) and "_type" in inner):
        return False
    return True


# ---------------------------------------------------------------------------
# Text rendering (debug / fallback)
# ---------------------------------------------------------------------------


def render_tree_text(snapshot: Any, indent: int = 0, prefix: str = "") -> str:
    """Render a snapshot dict-tree as indented text with focus markers."""
    if isinstance(snapshot, list):
        return "\n".join(render_tree_text(s, indent, prefix) for s in snapshot)

    pad = " " * indent

    if snapshot is None:
        return f"{pad}{prefix}∅"
    if not isinstance(snapshot, dict) or "_type" not in snapshot:
        return f"{pad}{prefix}{snapshot!r}"

    view = _get_view(snapshot)

    # Transparent wrapper — render inner content directly
    if view["shape"] is None:
        label_field = view["label"]
        if label_field and label_field in snapshot:
            return render_tree_text(snapshot[label_field], indent, prefix)
        return f"{pad}{prefix}∅"

    # Container node — show border and render inner subtree
    if _is_container_snapshot(snapshot):
        type_name = snapshot.get("_type", "?")
        marker = "  ◀━━ CURRENT" if snapshot.get("_focused") else ""
        lines = [f"{pad}{prefix}[{view['shape']}] {type_name}{marker}"]
        content_field = view["content_field"]
        lines.append(render_tree_text(snapshot[content_field], indent + 4))
        return "\n".join(lines)

    label = _get_label(snapshot)
    marker = "  ◀━━ CURRENT" if snapshot.get("_focused") else ""
    lines = [f"{pad}{prefix}[{view['shape']}]({label}){marker}"]

    for f in view["edges"]:
        if f in snapshot:
            child_val = snapshot[f]
            if isinstance(child_val, list):
                for i, item in enumerate(child_val):
                    lines.append(render_tree_text(item, indent + 4, f"{f}[{i}]: "))
            else:
                lines.append(render_tree_text(child_val, indent + 4, f"{f}: "))

    for f in view["data"]:
        if f in snapshot:
            val = snapshot[f]
            if isinstance(val, dict) and "_type" in val:
                lines.append(render_tree_text(val, indent + 4, f"{f}: "))
            elif isinstance(val, list):
                lines.append(f"{pad}    {f}: {_format_label_value(val)}")
            else:
                lines.append(f"{pad}    {f}: {val!r}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree layout — assign (x, y) coordinates to every node
# ---------------------------------------------------------------------------


class LayoutNode:
    """Intermediate layout element — a container with content.

    Every non-primitive rendered element is a LayoutNode.  Content is either
    text (for leaf nodes like ``RBNode``) or a nested subtree (for containers
    like ``RedBlackTree``).  Structural connections (edges/arrows) are stored
    in *children*; content nesting is stored in *content_root*.
    """

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
                f"Missing layout for node type '{self.type_name}'. "
                "No default layout is allowed."
            )
        self.layout_spec: LayoutSpec = normalize_layout_spec(raw_layout)
        _require_layout(self.layout_spec.name)

        # Content model: "text" (leaf) or "subtree" (container).
        # For leaf nodes, label holds the display text.
        # For containers, content_root holds the inner layout tree.
        self.content_type: str = "text"
        self.content_root: Optional[LayoutNode] = None
        self.type_label: str = ""

        # Collect child sub-trees from edge fields only
        self.child_fields: list[str] = []
        self.children: list[Optional[LayoutNode]] = []
        self.edge_styles: list[dict] = []
        for edge_spec in view["edge_specs"]:
            field_name = edge_spec["field"]
            if field_name not in snapshot:
                continue
            child = snapshot[field_name]
            if isinstance(child, list):
                # List of children → multiple edges from this node
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
                self.children.append(None)  # leaf / None child


def layout_tree(snapshot: Any) -> Optional[LayoutNode]:
    """Build a LayoutNode tree from a snapshot and assign coordinates.

    Transparent wrapper nodes (``shape=None``) are automatically unwrapped.
    Container nodes (``content_field`` is a registered subtree) produce a
    LayoutNode with ``content_type="subtree"`` that wraps the inner tree.
    """
    snapshot = _resolve_snapshot(snapshot)
    if snapshot is None or not isinstance(snapshot, dict) or "_type" not in snapshot:
        return None

    # Container: layout inner subtree and wrap in a container LayoutNode.
    if _is_container_snapshot(snapshot):
        view = _get_view(snapshot)
        inner = snapshot[view["content_field"]]
        inner_root = layout_tree(inner)  # recursive
        if inner_root is None:
            return None
        # Create container LayoutNode wrapping the inner tree
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
                f"Missing layout for node type '{container.type_name}'. "
                "No default layout is allowed."
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


# --- layout with direction hints ---


def _assign_positions(node: LayoutNode, depth: int) -> None:
    """Assign node coordinates by dispatching to explicit layout plugins."""
    node.x = 0.0
    node.y = -depth * V_GAP

    def _layout_for_edge(n: LayoutNode, spec: dict) -> LayoutSpec:
        raw = spec.get("layout")
        if raw is not None:
            return normalize_layout_spec(raw)
        if n.layout_spec is None:
            raise ValueError(
                f"Node '{n.type_name}' has no layout and edge '{spec.get('field')}' "
                "does not provide one."
            )
        return n.layout_spec

    def walk(n: LayoutNode, gap_x: float) -> None:
        real_pairs: list[tuple[LayoutNode, dict]] = [
            (c, spec)
            for c, spec in zip(n.children, n.edge_styles)
            if c is not None
        ]
        if not real_pairs:
            return

        specs = [_layout_for_edge(n, spec) for _, spec in real_pairs]
        key_set = {(s.name, json.dumps(s.params, sort_keys=True)) for s in specs}
        if len(key_set) != 1:
            raise ValueError(
                f"Node '{n.type_name}' has mixed child layouts. "
                "Use one layout per node level."
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
                f"Layout '{layout_spec.name}' must return LayoutResult, "
                f"got {type(result).__name__}"
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
                    f"Layout '{layout_spec.name}' returned invalid coordinate for "
                    f"child id {child.node_id}: {dx_dy!r}"
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
    """Shift so the tree is horizontally centered at x=0."""
    xs = list(_collect_x(root))
    if not xs:
        return
    cx = (min(xs) + max(xs)) / 2.0
    _shift_x(root, -cx)


def _clamp_width(root: LayoutNode) -> None:
    """Scale x-coordinates so the tree fits within MAX_SCENE_WIDTH."""
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
    """Collect all layout nodes into a flat list."""
    if node is None:
        return []
    result = [node]
    for c in node.children:
        result.extend(_flatten_nodes(c))
    return result


def _collect_render_info(
    root: LayoutNode,
    origin: np.ndarray,
) -> dict[int, _NodeRenderInfo]:
    """Collect ``{node_id: _NodeRenderInfo}`` from the layout tree.

    Both leaf nodes and container borders are emitted into the same dict,
    enabling unified diff-based animation.
    """
    info: dict[int, _NodeRenderInfo] = {}

    def walk(n: LayoutNode) -> None:
        if n.node_id is not None:
            if n.content_type == "subtree" and n.content_root is not None:
                # Container: emit a border element sized to its content
                x_min, x_max, y_min, y_max = _compute_bounding_box(
                    n.content_root, origin
                )
                pad = CONTAINER_PADDING
                info[n.node_id] = _NodeRenderInfo(
                    node_id=n.node_id,
                    pos=np.array(
                        [(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0]
                    ),
                    shape=n.shape,
                    fill_color=n.fill_color,
                    focused=n.focused,
                    width=max(x_max - x_min + pad * 2, 1.2),
                    height=max(y_max - y_min + pad * 2, 0.9),
                    type_label=n.type_label,
                    z_index=-2,
                )
                # Recurse into content subtree
                walk(n.content_root)
            else:
                # Leaf node: emit with text content
                info[n.node_id] = _NodeRenderInfo(
                    node_id=n.node_id,
                    pos=_node_pos(n, origin),
                    shape=n.shape,
                    fill_color=n.fill_color,
                    focused=n.focused,
                    text=n.label,
                )
        # Walk edge children
        for c in n.children:
            if c is not None:
                walk(c)

    walk(root)
    return info


def _collect_layout_edges(
    root: LayoutNode,
) -> dict[tuple[int, int], str]:
    """Collect ``{(parent_id, child_id): edge_style}`` from a layout tree."""
    edges: dict[tuple[int, int], str] = {}

    def walk(n: LayoutNode) -> None:
        # Walk into subtree content
        if n.content_type == "subtree" and n.content_root is not None:
            walk(n.content_root)
        for c, spec in zip(n.children, n.edge_styles):
            if c is not None and n.node_id is not None and c.node_id is not None:
                edges[(n.node_id, c.node_id)] = spec.get("style", "solid")
                walk(c)

    walk(root)
    return edges


# ---------------------------------------------------------------------------
# Manim Mobject builders
# ---------------------------------------------------------------------------


def _node_pos(node: LayoutNode, origin: np.ndarray) -> np.ndarray:
    return origin + np.array([node.x * H_GAP, node.y, 0.0])


def _node_radius(shape: str) -> float:
    """Return the effective radius / half-size for edge shortening."""
    if shape == "box":
        return BOX_WIDTH / 2
    if shape == "diamond":
        return NODE_RADIUS * 1.2
    return NODE_RADIUS


def _boundary_offset(shape: str, direction_unit: np.ndarray) -> np.ndarray:
    """Return offset from node center to boundary along *direction_unit*."""
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
        # |x| + |y| = r for a 45-degree diamond with vertical/horizontal radius r
        r = NODE_RADIUS * 1.2
        denom = abs(dx) + abs(dy)
        if denom < 1e-9:
            return np.array([0.0, 0.0, 0.0])
        t = r / denom
        return np.array([dx * t, dy * t, 0.0])

    # circle and fallback
    r = _node_radius(shape)
    return np.array([dx * r, dy * r, 0.0])


CONTAINER_PADDING = 0.55
CONTAINER_STROKE = 1.8
CONTAINER_FILL_OPACITY = 0.06
CONTAINER_LABEL_SIZE = 16


def _compute_bounding_box(
    root: LayoutNode, origin: np.ndarray
) -> tuple[float, float, float, float]:
    """Compute (x_min, x_max, y_min, y_max) from all nodes in a layout tree."""
    all_nodes = _flatten_nodes(root)
    positions = [_node_pos(n, origin) for n in all_nodes]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return min(xs), max(xs), min(ys), max(ys)


def _make_node_mob(info: _NodeRenderInfo) -> VGroup:
    """Create a visual element from render info.

    Both leaf nodes (shape + text label) and container borders (shape
    sized to content bounding box) flow through the same code path.
    When ``info.width``/``info.height`` are set, the shape is explicitly
    sized (container); otherwise default sizes are used (leaf).
    """
    has_explicit_size = info.width is not None

    # --- determine colors ---
    if info.fill_color:
        fill = ManimColor(info.fill_color)
        stroke_c = ManimColor(info.fill_color)
        if has_explicit_size:
            opacity = CONTAINER_FILL_OPACITY
            stroke_w = CONTAINER_STROKE
        else:
            opacity = 0.85
            stroke_w = FOCUS_STROKE if info.focused else 2.5
        txt_color = _contrast_text_color(info.fill_color)
        if info.focused and not has_explicit_size:
            stroke_c = FOCUS_COLOR
            stroke_w = FOCUS_STROKE + 1.0
    elif info.focused:
        fill, stroke_c = FOCUS_COLOR, FOCUS_COLOR
        opacity = FOCUS_FILL_OPACITY
        stroke_w = FOCUS_STROKE
        txt_color = WHITE
    else:
        if has_explicit_size:
            fill, stroke_c = GREY_B, GREY_B
            opacity = CONTAINER_FILL_OPACITY
            stroke_w = CONTAINER_STROKE
        else:
            fill, stroke_c = NORMAL_FILL, NORMAL_FILL
            opacity = NORMAL_FILL_OPACITY
            stroke_w = 2.0
        txt_color = WHITE

    # --- build shape (same branches handle both leaf and container sizes) ---
    pos = info.pos

    if info.shape == "diamond":
        r = max(info.width, info.height) / 2 if has_explicit_size else NODE_RADIUS * 1.2
        body = Polygon(
            pos + np.array([0, r, 0]),
            pos + np.array([r, 0, 0]),
            pos + np.array([0, -r, 0]),
            pos + np.array([-r, 0, 0]),
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )
    elif info.shape == "circle":
        radius = max(info.width, info.height) / 2 if has_explicit_size else NODE_RADIUS
        body = Circle(
            radius=radius,
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )
    else:  # "box" and fallback
        body = Rectangle(
            width=info.width or BOX_WIDTH,
            height=info.height or BOX_HEIGHT,
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )

    body.move_to(pos)
    body.set_z_index(info.z_index)

    parts: list[VMobject] = [body]

    # --- text content (leaf nodes) ---
    if info.text:
        font_size = 18 if info.shape == "diamond" else 20
        txt = Text(str(info.text), font="Menlo", font_size=font_size, color=txt_color)
        txt.move_to(pos)
        parts.append(txt)

    # --- type label at corner (containers) ---
    if info.type_label:
        lbl_color = _contrast_text_color(info.fill_color) if info.fill_color else GREY_A
        lbl = Text(
            info.type_label,
            font="Menlo",
            font_size=CONTAINER_LABEL_SIZE,
            color=lbl_color,
        )
        lbl.move_to(
            body.get_corner(UP + np.array([-1, 0, 0]))
            + np.array([lbl.width / 2 + 0.12, -0.18, 0])
        )
        lbl.set_z_index(info.z_index)
        parts.append(lbl)

    return VGroup(*parts)


def _make_edge(
    p1: np.ndarray,
    p2: np.ndarray,
    style: str = "solid",
    shape1: str = "circle",
    shape2: str = "circle",
) -> Line:
    """Create an edge between two node centers, anchored to shape boundaries."""
    d = p2 - p1
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-6:
        return Line(p1, p2, color=EDGE_COLOR, stroke_width=0)
    dn = d / d_len
    start = p1 + _boundary_offset(shape1, dn)
    end = p2 - _boundary_offset(shape2, dn)

    if style == "none":
        return Line(start, end, color=EDGE_COLOR, stroke_width=0)

    # Arrow for all edges
    edge = Arrow(
        start,
        end,
        buff=0,
        stroke_width=EDGE_STROKE,
        color=EDGE_COLOR,
        max_tip_length_to_length_ratio=0.15,
        max_stroke_width_to_length_ratio=8,
    )
    if style == "dashed":
        edge.set_stroke(opacity=0.7)
    elif style == "dotted":
        edge.set_stroke(opacity=0.5)
    return edge


# ---------------------------------------------------------------------------
# ManimStateRenderer — diff-based scene driver
# ---------------------------------------------------------------------------


class _ManimStateRenderer:
    """
    Diff-based renderer: maintains persistent mobjects keyed by node ``_id``
    and only animates the changes between consecutive states.

    Every visual element — both leaf nodes and container borders — is
    tracked in the same ``_node_mobs`` dict, enabling uniform diffing.
    """

    def __init__(
        self,
        scene: Scene,
        origin: np.ndarray | None = None,
        config: RenderConfig | None = None,
    ):
        self.scene = scene
        self.origin = origin if origin is not None else np.array([0.0, 0.5, 0.0])
        self.config = config or RenderConfig()
        self._node_mobs: dict[int, VGroup] = {}
        self._edge_mobs: dict[tuple[int, int], Line] = {}
        self._overlay: list[VMobject] = []

    def show_state(
        self, snapshot: Any, loc_text: str = "", counter_text: str = ""
    ) -> None:
        """Transition to a new state with minimal, diff-based animations."""
        snapshots = snapshot if isinstance(snapshot, list) else [snapshot]

        # Layout and collect all render info uniformly
        new_info: dict[int, _NodeRenderInfo] = {}
        new_edges: dict[tuple[int, int], str] = {}
        for snap in snapshots:
            lr = layout_tree(snap)
            if lr is not None:
                new_info.update(_collect_render_info(lr, self.origin))
                new_edges.update(_collect_layout_edges(lr))

        anims: list = []

        # Remove previous overlay (loc label, counter)
        for m in self._overlay:
            anims.append(FadeOut(m))
        self._overlay.clear()

        if not new_info:
            for m in self._node_mobs.values():
                anims.append(FadeOut(m))
            for m in self._edge_mobs.values():
                anims.append(FadeOut(m))
            self._node_mobs.clear()
            self._edge_mobs.clear()
            if anims:
                self.scene.play(*anims, run_time=ANIM_DURATION * 0.4)
            return

        old_nids = set(self._node_mobs)
        new_nids = set(new_info)
        old_eids = set(self._edge_mobs)
        new_eids = set(new_edges)

        # --- Remove vanished nodes / edges ---
        for nid in old_nids - new_nids:
            anims.append(FadeOut(self._node_mobs.pop(nid)))
        for eid in old_eids - new_eids:
            anims.append(FadeOut(self._edge_mobs.pop(eid)))

        # --- Transform persistent nodes ---
        for nid in old_nids & new_nids:
            target = _make_node_mob(new_info[nid])
            anims.append(ReplacementTransform(self._node_mobs[nid], target))
            self._node_mobs[nid] = target

        # --- Transform persistent edges ---
        for eid in old_eids & new_eids:
            pid, cid = eid
            style = new_edges[eid]
            p_info, c_info = new_info[pid], new_info[cid]
            new_edge = _make_edge(
                p_info.pos, c_info.pos, style, p_info.shape, c_info.shape
            )
            new_edge.set_z_index(-1)
            anims.append(ReplacementTransform(self._edge_mobs[eid], new_edge))
            self._edge_mobs[eid] = new_edge

        # --- FadeIn new nodes / edges ---
        for nid in new_nids - old_nids:
            mob = _make_node_mob(new_info[nid])
            self._node_mobs[nid] = mob
            anims.append(FadeIn(mob))
        for eid in new_eids - old_eids:
            pid, cid = eid
            style = new_edges[eid]
            p_info, c_info = new_info[pid], new_info[cid]
            edge = _make_edge(
                p_info.pos, c_info.pos, style, p_info.shape, c_info.shape
            )
            edge.set_z_index(-1)
            self._edge_mobs[eid] = edge
            anims.append(FadeIn(edge))

        # --- Overlay ---
        tc = self.config.text_color
        if isinstance(tc, str) and tc == "auto":
            overlay_color = GREY_A
        else:
            overlay_color = tc or GREY_A
        if loc_text:
            loc_mob = Text(loc_text, font="Menlo", font_size=16, color=overlay_color)
            loc_mob.to_edge(DOWN, buff=0.3)
            self._overlay.append(loc_mob)
            anims.append(FadeIn(loc_mob))

        if counter_text:
            cm = Text(counter_text, font="Menlo", font_size=14, color=overlay_color)
            cm.to_corner(DOWN + RIGHT * 0.1, buff=0.25)
            self._overlay.append(cm)
            anims.append(FadeIn(cm))

        if anims:
            self.scene.play(*anims, run_time=ANIM_DURATION)

    def clear(self) -> None:
        anims: list = []
        for m in self._node_mobs.values():
            anims.append(FadeOut(m))
        for m in self._edge_mobs.values():
            anims.append(FadeOut(m))
        for m in self._overlay:
            anims.append(FadeOut(m))
        if anims:
            self.scene.play(*anims, run_time=ANIM_DURATION * 0.4)
        self._node_mobs.clear()
        self._edge_mobs.clear()
        self._overlay.clear()


# ---------------------------------------------------------------------------
# Public API — render a list of State objects to video
# ---------------------------------------------------------------------------


def render_states(
    states: list,
    path: str,
    fps: int = 30,
    title: str = "",
    config: RenderConfig | None = None,
) -> Path:
    """
    Render a sequence of ``State`` objects into a manim media file.

    Generates a temporary manim Scene script, invokes ``manim`` CLI, and
    writes the result to *path*.  Returns the resolved output path.

    Parameters
    ----------
    states : list[State]
        The state sequence (from ``StateMachine.states``).
    path : str
        Output media path (e.g. ``"bst_search.gif"``).
    fps : int
        Frames per second.
    title : str
        Optional title shown at the top of the video.
    config : RenderConfig | None
        Rendering options (background color, quality, etc.).
    """
    cfg = config or RenderConfig()
    out = Path(path).resolve()
    logger.info("render_states: %d states -> %s", len(states), out)
    format_ext = out.suffix.lower().lstrip(".") or "mp4"
    if format_ext == "mov":
        format_ext = "mp4"
    output_name = out.stem if out.suffix else out.name

    # Build the scene class source dynamically so we can invoke manim CLI
    scene_src = _generate_scene_source(states, title=title, config=cfg)

    with tempfile.TemporaryDirectory(prefix="promin_") as tmp:
        script = Path(tmp) / "_promin_scene.py"
        script.write_text(scene_src, encoding="utf-8")

        quality_flag = f"-q{cfg.quality}"
        cmd = [
            sys.executable,
            "-m",
            "manim",
            "render",
            quality_flag,
            "--format",
            format_ext,
            "--fps",
            str(fps),
            "--media_dir",
            tmp,
            "-o",
            output_name,
            str(script),
            "ProminScene",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("render_states: manim stderr: %s", result.stderr)
            raise RuntimeError(f"manim render failed (exit {result.returncode})")

        # manim puts the file somewhere under media_dir — find it
        candidates = list(Path(tmp).rglob(f"{output_name}.{format_ext}"))
        if not candidates:
            candidates = list(Path(tmp).rglob(out.name))
        if candidates:
            import shutil

            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidates[0]), str(out))
            logger.info("render_states: moved output to %s", out)
        else:
            raise RuntimeError(
                f"manim render succeeded but output file was not found for {out.name}"
            )

    return out


def _generate_scene_source(
    states: list, title: str = "", config: RenderConfig | None = None
) -> str:
    """Generate a self-contained Python source that defines ProminScene."""
    import json

    cfg = config or RenderConfig()

    # Serialize states into JSON-safe dicts
    frames = []
    for s in states:
        frames.append(
            {
                "snapshot": s.snapshot,
                "loc": repr(s.current_loc) if s.current_loc else None,
            }
        )
    frames_json = json.dumps(frames, default=str)
    layout_bootstrap = _custom_layout_bootstrap_code()

    # Resolve adaptive colors
    bg = cfg.background_color or "#000000"
    if cfg.text_color == "auto":
        text_color = _contrast_text_color(bg)
    else:
        text_color = cfg.text_color or "#888888"
    title_color = cfg.title_color or YELLOW_C
    edge_color = cfg.edge_color or ""

    # Background setup line
    bg_line = ""
    if cfg.background_color:
        bg_line = f'self.camera.background_color = "{cfg.background_color}"'

    header = [
        "from __future__ import annotations",
        "import json",
        "import importlib",
        "import promin as pm",
        "from manim import *",
        "from promin.render import (",
        "    _ManimStateRenderer,",
        "    RenderConfig,",
        "    register_layout,",
        ")",
        "",
        f"FRAMES = json.loads({frames_json!r})",
    ]
    if layout_bootstrap:
        header.extend(["", layout_bootstrap])
    header.append("")

    scene_src = textwrap.dedent(
        f"""\
        class ProminScene(Scene):
            def construct(self):
                {bg_line}
                title_text = {title!r}
                title_color = {title_color!r}
                if title_text:
                    t = Text(title_text, font="Menlo", font_size=30, color=title_color)
                    t.to_edge(UP, buff=0.3)
                    self.play(FadeIn(t, shift=DOWN * 0.15), run_time=0.3)

                cfg = RenderConfig(
                    background_color={cfg.background_color!r},
                    node_color={cfg.node_color!r},
                    edge_color={cfg.edge_color!r},
                    title_color={cfg.title_color!r},
                    text_color={text_color!r},
                    quality={cfg.quality!r},
                )
                renderer = _ManimStateRenderer(self, config=cfg)
                n = len(FRAMES)
                for i, frame in enumerate(FRAMES):
                    loc = frame.get("loc")
                    loc_text = f"S{{i}}  {{loc}}" if loc else ""
                    counter = f"{{i+1}}/{{n}}"
                    renderer.show_state(
                        frame["snapshot"], loc_text=loc_text, counter_text=counter,
                    )
                    self.wait(0.3)

                renderer.clear()
                self.wait(1.0)
        """
    )
    return "\n".join(header) + scene_src


# ---------------------------------------------------------------------------
# In-process rendering (no subprocess) — for programmatic use
# ---------------------------------------------------------------------------


def render_states_inline(
    scene: Scene,
    states: list,
    title: str = "",
    origin: np.ndarray | None = None,
    config: RenderConfig | None = None,
) -> None:
    """
    Render states directly into an existing manim Scene (for embedding).

    Use this when you have your own Scene subclass and want to embed
    the state animation as part of a larger video.
    """
    cfg = config or RenderConfig()
    if origin is None:
        origin = np.array([0.0, 0.5, 0.0])

    logger.info(
        "render_states_inline: %d states, origin=%s", len(states), origin.tolist()
    )

    if cfg.background_color:
        scene.camera.background_color = cfg.background_color

    title_color = cfg.title_color or YELLOW_C
    if title:
        t = Text(title, font="Menlo", font_size=30, color=title_color)
        t.to_edge(UP, buff=0.3)
        scene.play(FadeIn(t, shift=DOWN * 0.15), run_time=0.3)

    renderer = _ManimStateRenderer(scene, origin=origin, config=cfg)
    n = len(states)
    for i, state in enumerate(states):
        loc_text = f"S{i}  {state.current_loc}" if state.current_loc else ""
        counter = f"{i + 1}/{n}"
        renderer.show_state(state.snapshot, loc_text=loc_text, counter_text=counter)
        scene.wait(0.3)

    renderer.clear()
    scene.wait(1.0)
