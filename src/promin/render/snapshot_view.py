from __future__ import annotations

from typing import Any, Optional

from manim import BLACK as MANIM_BLACK, WHITE

from ..view import View


def _get_view(snapshot: dict) -> dict:
    raw = snapshot.get("_view", {})
    container = raw.get("container", {})
    links = raw.get("links", {})
    items = links.get("items", [])
    return {
        "container": {
            "shape": container.get("shape"),
            "layout": container.get("layout"),
            "color_field": container.get("color_field", ""),
            "color_map": container.get("color_map", {}),
            "content_field": container.get("content_field", ""),
        },
        "links": {
            "layout": links.get("layout"),
            "items": items,
        },
        "shape": container.get("shape"),
        "link_fields": [item.get("field") for item in items],
        "link_specs": items,
        "content_field": container.get("content_field", ""),
    }


def _get_label(snapshot: dict) -> str:
    view = _get_view(snapshot)
    content_field = view["content_field"]
    if content_field and content_field in snapshot:
        return _format_label_value(snapshot[content_field], layout=view["container"].get("layout"))
    return str(snapshot.get("_type", "?"))


def _get_node_color(snapshot: dict) -> Optional[str]:
    view = _get_view(snapshot)
    c = view["container"]
    color_field = c["color_field"]
    if not color_field:
        return None
    field_value = snapshot.get(color_field)
    if field_value is None:
        return None
    color_map = c["color_map"]
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


def _format_label_value(val: Any, layout: Any = None) -> str:
    if isinstance(val, list):
        # container.content list means "multiple displayed fields", not list-view semantics
        parts = [_format_label_value(item, layout=None) for item in val]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        kind = str(getattr(layout, "_promin_layout_kind", ""))
        if "row" in kind:
            return " | ".join(parts)
        return "\n".join(parts)
    if isinstance(val, dict) and "_type" in val:
        inner_view = _get_view(val)
        inner_content = inner_view["content_field"]
        if inner_content and inner_content in val:
            return _format_label_value(
                val[inner_content], layout=inner_view["container"].get("layout")
            )
        return str(val.get("_type", "?"))
    return View.format_value(val)


def _resolve_snapshot(snapshot: Any) -> Any:
    if not isinstance(snapshot, dict) or "_type" not in snapshot:
        return snapshot
    view = _get_view(snapshot)
    if view["shape"] is not None:
        return snapshot
    # shape=None still keeps container semantics when content includes
    # structured nodes (e.g. ["root", tree.root]).
    if _first_structured_content_item(snapshot) is not None:
        return snapshot
    content_field = view["content_field"]
    # fallback for old snapshots that still carry "label"
    if not content_field:
        content_field = view["container"].get("label", "")
    if content_field and content_field in snapshot:
        inner = snapshot[content_field]
        return _resolve_snapshot(inner)
    return None


def _is_container_snapshot(snapshot: dict) -> bool:
    view = _get_view(snapshot)
    content_field = view["content_field"]
    if not content_field or content_field not in snapshot:
        return False
    inner = snapshot[content_field]
    if isinstance(inner, dict) and "_type" in inner:
        return True
    if isinstance(inner, list):
        return any(isinstance(item, dict) and "_type" in item for item in inner)
    return False


def _first_structured_content_item(snapshot: dict) -> Optional[dict]:
    view = _get_view(snapshot)
    content_field = view["content_field"]
    if not content_field or content_field not in snapshot:
        return None
    inner = snapshot[content_field]
    if isinstance(inner, dict) and "_type" in inner:
        return inner
    if isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and "_type" in item:
                return item
    return None


def _container_inline_label(snapshot: dict) -> str:
    """Text-only content entries in container.content (e.g. ['root', node])."""
    view = _get_view(snapshot)
    content_field = view["content_field"]
    if not content_field or content_field not in snapshot:
        return ""
    inner = snapshot[content_field]
    if isinstance(inner, list):
        parts: list[str] = []
        for item in inner:
            if isinstance(item, dict) and "_type" in item:
                continue
            formatted = _format_label_value(item, layout=view["container"].get("layout"))
            if formatted:
                parts.append(formatted)
        return "\n".join(parts)
    if isinstance(inner, dict) and "_type" in inner:
        return ""
    return _format_label_value(inner, layout=view["container"].get("layout"))


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
        content_field = view["content_field"]
        if content_field and content_field in snapshot:
            return render_tree_text(snapshot[content_field], indent, prefix)
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

    for spec in view["link_specs"]:
        f = spec["field"]
        if f in snapshot:
            child_val = snapshot[f]
            if isinstance(child_val, list):
                for i, item in enumerate(child_val):
                    lines.append(render_tree_text(item, indent + 4, f"{f}[{i}]: "))
            else:
                lines.append(render_tree_text(child_val, indent + 4, f"{f}: "))

    return "\n".join(lines)
