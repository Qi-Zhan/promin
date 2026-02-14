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
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .view import View

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

    label = _get_label(snapshot)
    marker = "  ◀━━ CURRENT" if snapshot.get("_focused") else ""
    view = _get_view(snapshot)
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
    """An intermediate node carrying position info during layout."""

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
                self.child_fields.append(field_name)
                self.edge_styles.append(edge_spec)
                self.children.append(LayoutNode(child))
            else:
                self.child_fields.append(field_name)
                self.edge_styles.append(edge_spec)
                self.children.append(None)  # leaf / None child


def layout_tree(snapshot: Any) -> Optional[LayoutNode]:
    """Build a LayoutNode tree from a snapshot and assign coordinates."""
    if snapshot is None or not isinstance(snapshot, dict) or "_type" not in snapshot:
        return None

    root = LayoutNode(snapshot)
    _assign_positions(root, depth=0)
    _center_tree(root)
    _clamp_width(root)
    return root


# --- layout with direction hints ---


def _assign_positions(node: LayoutNode, depth: int) -> None:
    """DFS inorder walk — assigns x = inorder slot, y = -depth.

    Respects edge direction hints: ``"left"`` children are placed before
    the parent, ``"right"`` after.  ``"auto"`` splits evenly.
    """
    _slot = [0.0]

    def walk(n: LayoutNode, d: int):
        left_children: list[LayoutNode] = []
        right_children: list[LayoutNode] = []

        for c, spec in zip(n.children, n.edge_styles):
            if c is None:
                continue
            direction = spec.get("direction", "auto")
            if direction == "left":
                left_children.append(c)
            elif direction == "right":
                right_children.append(c)
            else:
                # "auto" / "down" / "up" → split evenly
                left_children.append(c)  # will be rebalanced below

        # For "auto": split children roughly in half
        if all(
            spec.get("direction", "auto") in ("auto", "down", "up")
            for spec in n.edge_styles
        ):
            all_real = [c for c in n.children if c is not None]
            mid = len(all_real) // 2
            left_children = all_real[:mid]
            right_children = all_real[mid:]

        for c in left_children:
            walk(c, d + 1)

        n.x = _slot[0]
        n.y = -d * V_GAP
        _slot[0] += 1

        for c in right_children:
            walk(c, d + 1)

    walk(node, depth)


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


def _collect_node_info(
    root: LayoutNode,
    origin: np.ndarray,
) -> dict[int, tuple[str, np.ndarray, bool, str, Optional[str]]]:
    """Collect ``{node_id: (label, position, focused, shape, fill_color)}`` from layout."""
    info: dict[int, tuple[str, np.ndarray, bool, str, Optional[str]]] = {}

    def walk(n: LayoutNode) -> None:
        if n.node_id is not None:
            info[n.node_id] = (n.label, _node_pos(n, origin), n.focused, n.shape, n.fill_color)
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


def _make_node_mob(
    label: str,
    pos: np.ndarray,
    focused: bool = False,
    shape: str = "circle",
    fill_color: Optional[str] = None,
) -> VGroup:
    """Create a shape + label for a node.

    Parameters
    ----------
    fill_color : str | None
        If given, a hex color string (e.g. ``"#CC0000"``) used as the
        node fill.  Text color is automatically chosen for contrast.
    """
    # Determine colors
    if fill_color:
        fill = ManimColor(fill_color)
        stroke_c = ManimColor(fill_color)
        opacity = 0.85
        stroke_w = FOCUS_STROKE if focused else 2.5
        txt_color = _contrast_text_color(fill_color)
        if focused:
            stroke_c = FOCUS_COLOR
            stroke_w = FOCUS_STROKE + 1.0
    elif focused:
        fill, stroke_c = FOCUS_COLOR, FOCUS_COLOR
        opacity = FOCUS_FILL_OPACITY
        stroke_w = FOCUS_STROKE
        txt_color = WHITE
    else:
        fill, stroke_c = NORMAL_FILL, NORMAL_FILL
        opacity = NORMAL_FILL_OPACITY
        stroke_w = 2.0
        txt_color = WHITE

    if shape == "box":
        body = Rectangle(
            width=BOX_WIDTH,
            height=BOX_HEIGHT,
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )
    elif shape == "diamond":
        r = NODE_RADIUS * 1.2
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
        txt = Text(str(label), font="Menlo", font_size=18, color=txt_color)
        return VGroup(body, txt).move_to(pos)
    else:  # "circle" and fallback
        body = Circle(
            radius=NODE_RADIUS,
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )

    txt = Text(str(label), font="Menlo", font_size=20, color=txt_color)
    return VGroup(body, txt).move_to(pos)


def _make_edge(
    p1: np.ndarray,
    p2: np.ndarray,
    style: str = "solid",
    shape1: str = "circle",
    shape2: str = "circle",
) -> Line:
    """Create an edge line between two node centers, shortened by node radii."""
    d = p2 - p1
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-6:
        return Line(p1, p2, color=EDGE_COLOR, stroke_width=0)
    dn = d / d_len
    r1 = _node_radius(shape1)
    r2 = _node_radius(shape2)

    # Arrow for all edges
    edge = Arrow(
        p1 + dn * r1,
        p2 - dn * r2,
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
    """

    def __init__(self, scene: Scene, origin: np.ndarray | None = None):
        self.scene = scene
        self.origin = origin if origin is not None else np.array([0.0, 0.5, 0.0])
        self._node_mobs: dict[int, VGroup] = {}
        self._edge_mobs: dict[tuple[int, int], Line] = {}
        self._overlay: list[VMobject] = []

    def show_state(
        self, snapshot: Any, loc_text: str = "", counter_text: str = ""
    ) -> None:
        """Transition to a new state with minimal, diff-based animations."""
        snapshots = snapshot if isinstance(snapshot, list) else [snapshot]

        # Layout all captured trees and merge node/edge info
        new_info: dict[int, tuple[str, np.ndarray, bool, str]] = {}
        new_edges: dict[tuple[int, int], str] = {}
        for snap in snapshots:
            lr = layout_tree(snap)
            if lr is not None:
                new_info.update(_collect_node_info(lr, self.origin))
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

        # --- Transform persistent nodes (position + focus style + shape) ---
        for nid in old_nids & new_nids:
            lbl, pos, focused, shp, fc = new_info[nid]
            target = _make_node_mob(lbl, pos, focused, shp, fill_color=fc)
            anims.append(ReplacementTransform(self._node_mobs[nid], target))
            self._node_mobs[nid] = target

        # --- Transform persistent edges (endpoints may have shifted) ---
        for eid in old_eids & new_eids:
            pid, cid = eid
            style = new_edges[eid]
            _, p1, _, s1, _ = new_info[pid]
            _, p2, _, s2, _ = new_info[cid]
            new_edge = _make_edge(p1, p2, style, s1, s2)
            new_edge.set_z_index(-1)
            anims.append(ReplacementTransform(self._edge_mobs[eid], new_edge))
            self._edge_mobs[eid] = new_edge

        # --- FadeIn new nodes / edges ---
        for nid in new_nids - old_nids:
            lbl, pos, focused, shp, fc = new_info[nid]
            mob = _make_node_mob(lbl, pos, focused, shp, fill_color=fc)
            self._node_mobs[nid] = mob
            anims.append(FadeIn(mob))
        for eid in new_eids - old_eids:
            pid, cid = eid
            style = new_edges[eid]
            _, p1, _, s1, _ = new_info[pid]
            _, p2, _, s2, _ = new_info[cid]
            edge = _make_edge(p1, p2, style, s1, s2)
            edge.set_z_index(-1)
            self._edge_mobs[eid] = edge
            anims.append(FadeIn(edge))

        # --- Overlay ---
        if loc_text:
            loc_mob = Text(loc_text, font="Menlo", font_size=16, color=GREY_A)
            loc_mob.to_edge(DOWN, buff=0.3)
            self._overlay.append(loc_mob)
            anims.append(FadeIn(loc_mob))

        if counter_text:
            cm = Text(counter_text, font="Menlo", font_size=14, color=GREY_A)
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
) -> Path:
    """
    Render a sequence of ``State`` objects into a manim video.

    Generates a temporary manim Scene script, invokes ``manim`` CLI, and
    writes the result to *path*.  Returns the resolved output path.

    Parameters
    ----------
    states : list[State]
        The state sequence (from ``StateMachine.states``).
    path : str
        Output video file path (e.g. ``"bst_search.mp4"``).
    fps : int
        Frames per second.
    title : str
        Optional title shown at the top of the video.
    """
    out = Path(path).resolve()
    logger.info("render_states: %d states -> %s", len(states), out)

    # Build the scene class source dynamically so we can invoke manim CLI
    scene_src = _generate_scene_source(states, title=title)

    with tempfile.TemporaryDirectory(prefix="promin_") as tmp:
        script = Path(tmp) / "_promin_scene.py"
        script.write_text(scene_src, encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "manim",
            "render",
            "-ql",  # low quality for speed; user can override
            "--fps",
            str(fps),
            "--media_dir",
            tmp,
            "-o",
            out.name,
            str(script),
            "ProminScene",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("render_states: manim stderr: %s", result.stderr)
            print("manim stderr:", result.stderr)
            raise RuntimeError(f"manim render failed (exit {result.returncode})")

        # manim puts the file somewhere under media_dir — find it
        candidates = list(Path(tmp).rglob(out.name))
        if candidates:
            import shutil

            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidates[0]), str(out))
            logger.info("render_states: moved output to %s", out)

    return out


def _generate_scene_source(states: list, title: str = "") -> str:
    """Generate a self-contained Python source that defines ProminScene."""
    import json

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

    return textwrap.dedent(
        f"""\
        import json
        from manim import *
        from promin.render import _ManimStateRenderer

        FRAMES = json.loads({frames_json!r})

        class ProminScene(Scene):
            def construct(self):
                title_text = {title!r}
                if title_text:
                    t = Text(title_text, font="Menlo", font_size=30, color=YELLOW_C)
                    t.to_edge(UP, buff=0.3)
                    self.play(FadeIn(t, shift=DOWN * 0.15), run_time=0.3)

                renderer = _ManimStateRenderer(self)
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


# ---------------------------------------------------------------------------
# In-process rendering (no subprocess) — for programmatic use
# ---------------------------------------------------------------------------


def render_states_inline(
    scene: Scene,
    states: list,
    title: str = "",
    origin: np.ndarray | None = None,
) -> None:
    """
    Render states directly into an existing manim Scene (for embedding).

    Use this when you have your own Scene subclass and want to embed
    the state animation as part of a larger video.
    """
    if origin is None:
        origin = np.array([0.0, 0.5, 0.0])

    logger.info(
        "render_states_inline: %d states, origin=%s", len(states), origin.tolist()
    )

    if title:
        t = Text(title, font="Menlo", font_size=30, color=YELLOW_C)
        t.to_edge(UP, buff=0.3)
        scene.play(FadeIn(t, shift=DOWN * 0.15), run_time=0.3)

    renderer = _ManimStateRenderer(scene, origin=origin)
    n = len(states)
    for i, state in enumerate(states):
        loc_text = f"S{i}  {state.current_loc}" if state.current_loc else ""
        counter = f"{i + 1}/{n}"
        renderer.show_state(state.snapshot, loc_text=loc_text, counter_text=counter)
        scene.wait(0.3)

    renderer.clear()
    scene.wait(1.0)
