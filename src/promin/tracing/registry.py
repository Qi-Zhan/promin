from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..view import (
    TypeViewSpec,
    EdgeSpec,
    LayoutSpec,
    normalize_edges,
    normalize_layout_spec,
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
            "Example: layout={'name': 'row', 'params': {}}"
        )
    info.view = view_spec
    return True


def _validate_layout(layout: Any) -> LayoutSpec:
    if isinstance(layout, str):
        raise TypeError(
            "layout must be a dict {'name': <str>, 'params': <dict>}, "
            "not a bare string."
        )
    spec = normalize_layout_spec(layout)
    if not spec.name:
        raise TypeError(
            "layout.name must be a non-empty string. "
            "Example: layout={'name': 'tree', 'params': {}}"
        )
    if not isinstance(spec.params, dict):
        raise TypeError("layout.params must be a dict")
    try:
        json.dumps(spec.params)
    except TypeError as exc:
        raise TypeError("layout.params must be JSON-serializable") from exc
    return spec


def register_type(
    cls: type | None = None,
    *,
    layout: LayoutSpec | dict[str, Any],
    shape: Optional[str] = "circle",
    label: str = "",
    edges: list[str | EdgeSpec] | None = None,
    data: list[str] | None = None,
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
        normalized_layout = _validate_layout(layout)
        if content_field is not None and (
            not isinstance(content_field, str) or not content_field
        ):
            raise TypeError("content_field must be a non-empty string when provided")
        view = TypeViewSpec(
            shape=shape,
            label=label,
            edges=normalize_edges(edges or []),
            data=data or [],
            color_field=color_field,
            color_map=color_map or {},
            layout=normalized_layout,
            content_field=content_field or "",
        )
        if view.content_field and view.content_field not in view.fields:
            raise TypeError(
                "content_field must reference a tracked field "
                "(label, edge, or data field)."
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
            "register_type: %s shape=%s label=%s edges=%d data=%d",
            name,
            shape,
            label,
            len(view.edges),
            len(view.data),
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
        layout={"name": "row", "params": {}},
        shape="box",
        label="value",
        data=["value"],
        type_name="int",
        label_resolver=lambda v: v,
        data_resolver=lambda v: {"value": v},
        focusable=False,
        register_view=False,
    )
    register_type(
        bool,
        layout={"name": "row", "params": {}},
        shape="diamond",
        label="value",
        data=["value"],
        type_name="bool",
        label_resolver=lambda v: v,
        data_resolver=lambda v: {"value": v},
        focusable=False,
        register_view=False,
    )
    register_type(
        list,
        layout={"name": "row", "params": {"wrap": True, "columns": 8}},
        shape="box",
        label="summary",
        edges=[
            EdgeSpec(
                field="elements",
                direction="right",
                layout=LayoutSpec(name="row", params={"wrap": True, "columns": 8}),
            )
        ],
        data=["summary"],
        type_name="list",
        label_resolver=lambda v: f"len={len(v)}",
        children_resolver=lambda v: {"elements": list(v)},
        data_resolver=lambda v: {"summary": f"len={len(v)}"},
        register_view=False,
    )


_register_builtin_types()
