from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .dsl import container, links
from .specs import TypeViewSpec
from .serialize import type_view_to_dict


@dataclass
class StyleContext:
    highlight: dict[str, Any] = field(default_factory=dict)
    color: Optional[str] = None
    label: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "StyleContext") -> "StyleContext":
        return StyleContext(
            highlight={**self.highlight, **other.highlight},
            color=other.color or self.color,
            label=other.label or self.label,
            extras={**self.extras, **other.extras},
        )


class View:
    _type_to_view: dict[type, Callable[[], "View"]] = {}

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        raise NotImplementedError

    def format_label(self, value: Any) -> str:
        return str(value)

    def type_view_spec(self) -> Optional[TypeViewSpec]:
        return None

    @staticmethod
    def register(value_type: type, view_factory: Callable[[], "View"]) -> None:
        View._type_to_view[value_type] = view_factory

    @staticmethod
    def for_value(value: Any) -> "View":
        for cls in type(value).__mro__:
            if cls in View._type_to_view:
                return View._type_to_view[cls]()
        raise TypeError(f"No view registered for {type(value).__name__}")

    @classmethod
    def format_value(cls, value: Any) -> str:
        if value is None:
            return NoneView().format_label(value)
        try:
            view = cls.for_value(value)
            return view.format_label(value)
        except TypeError:
            return repr(value)

    @classmethod
    def render_value(cls, value: Any, style: Optional[StyleContext] = None) -> Any:
        try:
            view = cls.for_value(value)
            return view.render(value, style)
        except TypeError:
            return {"type": "unknown", "value": repr(value), "style": style}


class StrView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "str", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)


class BoolView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "bool", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)

    def type_view_spec(self) -> Optional[TypeViewSpec]:
        return TypeViewSpec(
            container=container(shape="diamond", content=lambda v: [v]),
            links=links().build(),
        )


class NoneView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "none", "value": None, "style": style}

    def format_label(self, value: Any) -> str:
        return "\u2205"


class IntView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "int", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)

    def type_view_spec(self) -> Optional[TypeViewSpec]:
        return TypeViewSpec(
            container=container(shape="box", content=lambda v: [v]),
            links=links().build(),
        )


class FloatView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "float", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)


class ListView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "list", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [View.format_value(item) for item in value]
        return "[" + ", ".join(items) + "]"

    def type_view_spec(self) -> Optional[TypeViewSpec]:
        from ..layout import row

        return TypeViewSpec(
            container=container(shape="box", content=lambda v: [f"len={len(v)}"]),
            links=links().items(lambda v: list(v)).layout(row(wrap=True, columns=8)).build(),
        )


class DictView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = {k: View.render_value(v) for k, v in value.items()}
        return {"type": "dict", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [f"{k}: {View.format_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"


class TupleView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "tuple", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [View.format_value(item) for item in value]
        return "(" + ", ".join(items) + ")"


class SetView(View):
    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "set", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [View.format_value(item) for item in value]
        return "{" + ", ".join(items) + "}"


class RegisteredClassView(View):
    def __init__(self, type_view: TypeViewSpec):
        self.type_view = type_view

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {
            "type": "registered_node",
            "node_view": type_view_to_dict(self.type_view),
            "value": value,
            "style": style,
        }

    def format_label(self, value: Any) -> str:
        c = self.type_view.container.content
        if c is not None:
            content_values = c(value)
            if not isinstance(content_values, list):
                raise TypeError("container.content must return list")
            if len(content_values) == 1:
                return View.format_value(content_values[0])
            return "\n".join(View.format_value(v) for v in content_values)
        return type(value).__name__


View.register(type(None), lambda: NoneView())
View.register(bool, lambda: BoolView())
View.register(int, lambda: IntView())
View.register(float, lambda: FloatView())
View.register(str, lambda: StrView())
View.register(list, lambda: ListView())
View.register(dict, lambda: DictView())
View.register(tuple, lambda: TupleView())
View.register(set, lambda: SetView())
