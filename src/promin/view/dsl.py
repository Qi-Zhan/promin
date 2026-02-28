from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .specs import ContainerSpec, LinksSpec


def container(
    *,
    shape: str | None = None,
    layout: Callable[..., Any] | None = None,
    content: Callable[[Any], list[Any]] | None = None,
    color: Callable[[Any], Any] | None = None,
    text_color: Callable[[Any], Any] | None = None,
    color_map: dict[str, str] | None = None,
    text_color_map: dict[str, str] | None = None,
) -> ContainerSpec:
    return ContainerSpec(
        shape=shape,
        layout=layout,
        content=content,
        color=color,
        text_color=text_color,
        color_map=color_map or {},
        text_color_map=text_color_map or {},
    )


@dataclass
class LinksBuilder:
    _items_resolver: Callable[[Any], list[Any]] | None = None
    _hints_resolver: Callable[[Any], list[str]] | None = None
    _layout: Callable[..., Any] | None = None

    def items(self, resolver: Callable[[Any], list[Any]]) -> "LinksBuilder":
        if not callable(resolver):
            raise TypeError("links.items resolver must be callable")
        self._items_resolver = resolver
        return self

    def hints(self, value: list[str] | Callable[[Any], list[str]]) -> "LinksBuilder":
        if isinstance(value, list):
            hints_list = list(value)
            self._hints_resolver = lambda _obj, _h=hints_list: list(_h)
            return self
        if callable(value):
            self._hints_resolver = value
            return self
        raise TypeError("links.hints must be list[str] or callable")

    def layout(self, layout_fn: Callable[..., Any]) -> "LinksBuilder":
        if not callable(layout_fn):
            raise TypeError("links.layout must be callable")
        self._layout = layout_fn
        return self

    def build(self) -> LinksSpec:
        return LinksSpec(
            items_resolver=self._items_resolver,
            hints_resolver=self._hints_resolver,
            layout=self._layout,
        )


def links() -> LinksBuilder:
    return LinksBuilder()
