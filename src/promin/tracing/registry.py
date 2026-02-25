from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..render.layout_registry import RowLayout
from ..view import (
    TypeViewSpec,
    EdgeSpec,
    normalize_edges,
    View,
    RegisteredClassView,
)

logger = logging.getLogger(__name__)


@dataclass
class _TypeInfo:
    """Internal metadata for a registered type."""

    type_name: str
    view: TypeViewSpec
    skip_if: Optional[Callable] = None
    label_resolver: Optional[Callable[[Any], Any]] = None
    children_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    data_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    focusable: bool = True

    @property
    def fields(self) -> list[str]:
        return self.view.fields


_registered_types: dict[type, _TypeInfo] = {}


def override_type_view_spec(value_type: type, view_spec: TypeViewSpec) -> bool:
    info = _registered_types.get(value_type)
    if info is None:
        return False
    if view_spec.layout is None:
        raise TypeError(
            "TypeViewSpec.layout is required. "
            "Example: layout=pm.TreeLayout"
        )
    view_spec.layout = _validate_layout(view_spec.layout)
    for edge in view_spec.edges:
        if edge.layout is not None:
            edge.layout = _validate_layout(edge.layout)
    info.view = view_spec
    return True


def _validate_layout(layout: Any):
    if not callable(layout):
        raise TypeError("layout must be callable: layout=pm.TreeLayout or layout=my_layout")
    return layout


def register_type(
    cls: type | None = None,
    *,
    layout: Callable[..., Any],
    shape: Optional[str] = "circle",
    label: str = "",
    edges: list[str | EdgeSpec] | None = None,
    type_name: str = "",
    color_field: str = "",
    color_map: dict[str, str] | None = None,
    skip_if: Optional[Callable] = None,
    label_resolver: Optional[Callable[[Any], Any]] = None,
    children_resolver: Optional[Callable[[Any], dict[str, Any]]] = None,
    data_resolver: Optional[Callable[[Any], dict[str, Any]]] = None,
    content_field: str | None = None,
    focusable: bool = True,
    register_view: bool = True,
):
    def _register(target_cls: type):
        name = type_name or target_cls.__name__
        if not callable(layout):
            raise TypeError("layout must be callable: layout=pm.TreeLayout or layout=my_layout")
        normalized_layout = _validate_layout(layout)
        if content_field is not None and (
            not isinstance(content_field, str) or not content_field
        ):
            raise TypeError("content_field must be a non-empty string when provided")
        normalized_edges = normalize_edges(edges or [])
        for edge in normalized_edges:
            if edge.layout is not None:
                if not callable(edge.layout):
                    raise TypeError(
                        "EdgeSpec.layout must be callable: layout=pm.RowLayout(...) or layout=my_layout"
                    )
                edge.layout = _validate_layout(edge.layout)

        view = TypeViewSpec(
            shape=shape,
            label=label,
            edges=normalized_edges,
            color_field=color_field,
            color_map=color_map or {},
            layout=normalized_layout,
            content_field=content_field or "",
        )
        if view.content_field and view.content_field not in view.fields:
            raise TypeError(
                "content_field must reference a tracked field "
                "(label, edge, color_field, or content_field)."
            )
        _registered_types[target_cls] = _TypeInfo(
            type_name=name,
            view=view,
            skip_if=skip_if,
            label_resolver=label_resolver,
            children_resolver=children_resolver,
            data_resolver=data_resolver,
            focusable=focusable,
        )
        logger.info(
            "register_type: %s shape=%s label=%s edges=%d",
            name,
            shape,
            label,
            len(view.edges),
        )
        if register_view:
            View.register(target_cls, lambda v=view: RegisteredClassView(v))
        return target_cls

    if cls is not None:
        return _register(cls)

    def decorator(target_cls):
        return _register(target_cls)

    return decorator


def _register_builtin_types() -> None:
    register_type(
        int,
        layout=RowLayout(),
        shape="box",
        label="value",
        type_name="int",
        label_resolver=lambda v: v,
        data_resolver=lambda v: {"value": v},
        focusable=False,
        register_view=False,
    )
    register_type(
        bool,
        layout=RowLayout(),
        shape="diamond",
        label="value",
        type_name="bool",
        label_resolver=lambda v: v,
        data_resolver=lambda v: {"value": v},
        focusable=False,
        register_view=False,
    )
    register_type(
        list,
        layout=RowLayout(wrap=True, columns=8),
        shape="box",
        label="summary",
        edges=[
            EdgeSpec(
                field="elements",
                direction="right",
                layout=RowLayout(wrap=True, columns=8),
            )
        ],
        type_name="list",
        label_resolver=lambda v: f"len={len(v)}",
        children_resolver=lambda v: {"elements": list(v)},
        data_resolver=lambda v: {"summary": f"len={len(v)}"},
        register_view=False,
    )


_register_builtin_types()
