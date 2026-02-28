from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ContainerSpec:
    shape: str | None = None
    layout: Callable[..., Any] | None = None
    content: Callable[[Any], list[Any]] | None = None
    color: Callable[[Any], Any] | None = None
    text_color: Callable[[Any], Any] | None = None
    color_map: dict[str, str] = field(default_factory=dict)
    text_color_map: dict[str, str] = field(default_factory=dict)

    # compiled fields used in snapshots
    content_field_name: str = "__content"
    color_field_name: str = "__color"
    text_color_field_name: str = "__text_color"


@dataclass
class LinkSpec:
    field: str
    resolver: Callable[[Any], Any] | None = None
    hint: str = "auto"
    style: str = "solid"


@dataclass
class LinksSpec:
    # New canonical model: one list resolver + optional hints resolver.
    items_resolver: Callable[[Any], list[Any]] | None = None
    hints_resolver: Callable[[Any], list[str]] | None = None
    layout: Callable[..., Any] | None = None
    # Transitional compatibility field; may be removed after full migration.
    items: list[LinkSpec] = field(default_factory=list)


@dataclass
class TypeViewSpec:
    container: ContainerSpec
    links: LinksSpec = field(default_factory=LinksSpec)

    @property
    def fields(self) -> list[str]:
        raw: list[str] = []
        if self.container.content is not None:
            raw.append(self.container.content_field_name)
        if self.container.color is not None:
            raw.append(self.container.color_field_name)
        if self.container.text_color is not None:
            raw.append(self.container.text_color_field_name)
        if self.links.items_resolver is not None:
            raw.append("__links")
        raw.extend(link.field for link in self.links.items)
        return list(dict.fromkeys(raw))
