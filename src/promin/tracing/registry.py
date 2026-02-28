from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..layout import row
from ..view import (
    ContainerSpec,
    LinksBuilder,
    LinksSpec,
    RegisteredClassView,
    TypeViewSpec,
    View,
)

logger = logging.getLogger(__name__)


@dataclass
class _TypeInfo:
    type_name: str
    view: TypeViewSpec
    skip_if: Optional[Callable] = None
    children_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    data_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    focusable: bool = True

    @property
    def fields(self) -> list[str]:
        return self.view.fields


_registered_types: dict[type, _TypeInfo] = {}


def _validate_view(view_spec: TypeViewSpec) -> TypeViewSpec:
    if not isinstance(view_spec, TypeViewSpec):
        raise TypeError("view must be a TypeViewSpec instance")

    c = view_spec.container
    if c.shape is not None and not isinstance(c.shape, str):
        raise TypeError("container.shape must be str | None")
    if c.layout is not None and not callable(c.layout):
        raise TypeError("container.layout must be callable")
    if c.content is not None and not callable(c.content):
        raise TypeError("container.content must be callable")
    if c.color is not None and not callable(c.color):
        raise TypeError("container.color must be callable")
    if c.text_color is not None and not callable(c.text_color):
        raise TypeError("container.text_color must be callable")

    if view_spec.links.layout is not None and not callable(view_spec.links.layout):
        raise TypeError("links.layout must be callable")
    if view_spec.links.items_resolver is not None and not callable(view_spec.links.items_resolver):
        raise TypeError("links.items resolver must be callable")
    if view_spec.links.hints_resolver is not None and not callable(view_spec.links.hints_resolver):
        raise TypeError("links.hints resolver must be callable")
    for item in view_spec.links.items:
        if not isinstance(item.field, str) or not item.field:
            raise TypeError("link field must be a non-empty string")
        if item.resolver is not None and not callable(item.resolver):
            raise TypeError("link resolver must be callable")

    return view_spec


def _register_type_core(
    target_cls: type,
    *,
    view_spec: TypeViewSpec,
    type_name: str = "",
    skip_if: Optional[Callable] = None,
    children_resolver: Optional[Callable[[Any], dict[str, Any]]] = None,
    data_resolver: Optional[Callable[[Any], dict[str, Any]]] = None,
    focusable: bool = True,
    register_view: bool = True,
) -> type:
    name = type_name or target_cls.__name__
    view_spec = _validate_view(view_spec)

    def _auto_content_data(obj: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if view_spec.container.color is not None:
            out[view_spec.container.color_field_name] = view_spec.container.color(obj)
        if view_spec.container.text_color is not None:
            out[view_spec.container.text_color_field_name] = view_spec.container.text_color(obj)
        if view_spec.container.content is not None:
            content_val = view_spec.container.content(obj)
            if not isinstance(content_val, list):
                raise TypeError(
                    f"{name}.container.content must return list, got "
                    f"{type(content_val).__name__}"
                )
            out[view_spec.container.content_field_name] = content_val
        return out

    def _auto_content_children(obj: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if view_spec.links.items_resolver is not None:
            raw = view_spec.links.items_resolver(obj)
            if not isinstance(raw, list):
                raise TypeError(
                    f"{name}.links.items must return list, got {type(raw).__name__}"
                )
            hints: list[str] = []
            if view_spec.links.hints_resolver is not None:
                hints = view_spec.links.hints_resolver(obj)
                if not isinstance(hints, list):
                    raise TypeError(
                        f"{name}.links.hints must return list, got {type(hints).__name__}"
                    )
            packed: list[dict[str, Any]] = []
            for i, item in enumerate(raw):
                hint = hints[i] if i < len(hints) else "auto"
                packed.append({"target": item, "_hint": hint, "_style": "solid"})
            out["__links"] = packed
        for item in view_spec.links.items:
            if item.resolver is not None:
                out[item.field] = item.resolver(obj)
        return out

    effective_children_resolver = children_resolver
    if children_resolver is None:
        effective_children_resolver = _auto_content_children
    else:
        effective_children_resolver = lambda obj, _base=children_resolver: {
            **(_auto_content_children(obj) or {}),
            **(_base(obj) or {}),
        }

    effective_data_resolver = data_resolver
    if data_resolver is None:
        effective_data_resolver = _auto_content_data
    else:
        effective_data_resolver = lambda obj, _base=data_resolver: {
            **(_auto_content_data(obj) or {}),
            **(_base(obj) or {}),
        }

    _registered_types[target_cls] = _TypeInfo(
        type_name=name,
        view=view_spec,
        skip_if=skip_if,
        children_resolver=effective_children_resolver,
        data_resolver=effective_data_resolver,
        focusable=focusable,
    )
    logger.info(
        "register_type: %s shape=%s links=%d",
        name,
        view_spec.container.shape,
        (1 if view_spec.links.items_resolver is not None else 0) + len(view_spec.links.items),
    )
    if register_view:
        View.register(target_cls, lambda v=view_spec: RegisteredClassView(v))
    return target_cls


@dataclass
class TypeBuilder:
    explicit_name: str | None = None
    _container: ContainerSpec = field(default_factory=ContainerSpec)
    _links: LinksSpec = field(default_factory=LinksSpec)
    _skip_if: Optional[Callable] = None
    _children_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    _data_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    _focusable: bool = True
    _register_view: bool = True

    def __call__(self, target_cls: type) -> type:
        view_spec = TypeViewSpec(container=self._container, links=self._links)
        return _register_type_core(
            target_cls,
            view_spec=view_spec,
            type_name=self.explicit_name or "",
            skip_if=self._skip_if,
            children_resolver=self._children_resolver,
            data_resolver=self._data_resolver,
            focusable=self._focusable,
            register_view=self._register_view,
        )

    def named(self, name: str) -> "TypeBuilder":
        self.explicit_name = name
        return self

    def shape(self, value: str | None) -> "TypeBuilder":
        self._container.shape = value
        return self

    def show(self, resolver: Callable[[Any], list[Any]]) -> "TypeBuilder":
        self._container.content = resolver
        return self

    def fill(
        self,
        resolver: Callable[[Any], Any],
        *,
        map: Optional[dict[str, str]] = None,
    ) -> "TypeBuilder":
        self._container.color = resolver
        self._container.color_map = dict(map or {})
        return self

    def text(
        self,
        resolver: Callable[[Any], Any],
        *,
        map: Optional[dict[str, str]] = None,
    ) -> "TypeBuilder":
        self._container.text_color = resolver
        self._container.text_color_map = dict(map or {})
        return self

    def layout(self, layout: Callable[..., Any]) -> "TypeBuilder":
        self._container.layout = layout
        return self

    def links(self, config: LinksBuilder) -> "TypeBuilder":
        if not isinstance(config, LinksBuilder):
            raise TypeError("type().links(...) expects pm.links() builder")
        self._links = config.build()
        return self

    def skip_if(self, pred: Callable[[Any], bool]) -> "TypeBuilder":
        self._skip_if = pred
        return self

    def focusable(self, enabled: bool) -> "TypeBuilder":
        self._focusable = enabled
        return self

    def children(self, resolver: Callable[[Any], dict[str, Any]]) -> "TypeBuilder":
        self._children_resolver = resolver
        return self

    def data(self, resolver: Callable[[Any], dict[str, Any]]) -> "TypeBuilder":
        self._data_resolver = resolver
        return self

    def no_view_registration(self) -> "TypeBuilder":
        self._register_view = False
        return self

def type_builder(type_name: str | None = None) -> TypeBuilder:
    return TypeBuilder(explicit_name=type_name)


def _register_builtin_types() -> None:
    _register_type_core(
        int,
        view_spec=TypeViewSpec(
            container=ContainerSpec(shape="box", content=lambda v: [v]),
            links=LinksSpec(),
        ),
        type_name="int",
        data_resolver=lambda v: {"value": v},
        focusable=False,
        register_view=False,
    )
    _register_type_core(
        bool,
        view_spec=TypeViewSpec(
            container=ContainerSpec(shape="diamond", content=lambda v: [v]),
            links=LinksSpec(),
        ),
        type_name="bool",
        data_resolver=lambda v: {"value": v},
        focusable=False,
        register_view=False,
    )
    _register_type_core(
        list,
        view_spec=TypeViewSpec(
            container=ContainerSpec(shape="box", content=lambda v: [f"len={len(v)}"]),
            links=LinksSpec(
                items_resolver=lambda v: list(v),
                layout=row(wrap=True, columns=8),
            ),
        ),
        type_name="list",
        data_resolver=lambda v: {"summary": f"len={len(v)}"},
        register_view=False,
    )


_register_builtin_types()
