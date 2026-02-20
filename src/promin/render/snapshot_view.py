from __future__ import annotations

from typing import Any, Optional

from manim import BLACK as MANIM_BLACK, WHITE

from ..view import View


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
    view = _get_view(snapshot)
    label_field = view["label"]
    if label_field and label_field in snapshot:
        return _format_label_value(snapshot[label_field])
    return str(snapshot.get("_type", "?"))


def _get_node_color(snapshot: dict) -> Optional[str]:
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
    return str(field_value)


def _contrast_text_color(fill_hex: str) -> str:
    hex_str = fill_hex.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    try:
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    except (ValueError, IndexError):
        return WHITE
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return MANIM_BLACK if luminance > 0.5 else WHITE


def _format_label_value(val: Any) -> str:
    if isinstance(val, dict) and "_type" in val:
        inner_view = _get_view(val)
        inner_label = inner_view["label"]
        if inner_label and inner_label in val:
            return _format_label_value(val[inner_label])
        return str(val.get("_type", "?"))
    return View.format_value(val)


def _resolve_snapshot(snapshot: Any) -> Any:
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
    view = _get_view(snapshot)
    if view["shape"] is None:
        return False
    content_field = view["content_field"]
    if not content_field or content_field not in snapshot:
        return False
    inner = snapshot[content_field]
    if not (isinstance(inner, dict) and "_type" in inner):
        return False
    return True


def render_tree_text(snapshot: Any, indent: int = 0, prefix: str = "") -> str:
    if isinstance(snapshot, list):
        return "\n".join(render_tree_text(s, indent, prefix) for s in snapshot)

    pad = " " * indent

    if snapshot is None:
        return f"{pad}{prefix}∅"
    if not isinstance(snapshot, dict) or "_type" not in snapshot:
        return f"{pad}{prefix}{snapshot!r}"

    view = _get_view(snapshot)

    if view["shape"] is None:
        label_field = view["label"]
        if label_field and label_field in snapshot:
            return render_tree_text(snapshot[label_field], indent, prefix)
        return f"{pad}{prefix}∅"

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
