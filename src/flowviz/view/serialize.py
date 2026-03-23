from __future__ import annotations

from typing import Any

from .specs import TypeViewSpec


def type_view_to_dict(spec: TypeViewSpec) -> dict[str, Any]:
    content_field = spec.container.content_field_name if spec.container.content is not None else ""
    color_field = spec.container.color_field_name if spec.container.color is not None else ""
    text_color_field = spec.container.text_color_field_name if spec.container.text_color is not None else ""

    link_items: list[dict[str, Any]] = []
    if spec.links.items_resolver is not None:
        link_items.append({"field": "__links", "hint": "auto", "style": "solid"})
    link_items.extend(
        {
            "field": item.field,
            "hint": item.hint,
            "style": item.style,
        }
        for item in spec.links.items
    )

    return {
        "container": {
            "shape": spec.container.shape,
            "layout": spec.container.layout,
            "content_field": content_field,
            "color_field": color_field,
            "text_color_field": text_color_field,
            "color_map": dict(spec.container.color_map),
            "text_color_map": dict(spec.container.text_color_map),
        },
        "links": {
            "layout": spec.links.layout,
            "items": link_items,
        },
    }
