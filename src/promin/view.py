"""
promin.view — Declarative visual vocabulary for registered classes.

Users describe *how* to render a class via :func:`register_class` parameters:

* **shape** — ``"circle"``, ``"box"``, ``"diamond"``
* **label** — which field is shown inside the shape (e.g. ``"key"``)
* **edges** — which fields are connections to other nodes
* **data**  — extra tracked fields (not rendered as connections)

These specs are stored in :class:`NodeView` and embedded in snapshots so
the renderer can be fully data-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable


# ---------------------------------------------------------------------------
# Edge descriptor
# ---------------------------------------------------------------------------


@dataclass
class EdgeSpec:
    """Describes a field that connects to a child / related node.

    Parameters
    ----------
    field : str
        Attribute name on the class (must resolve to another registered object
        or ``None``).
    direction : str
        Layout hint for the edge.  ``"auto"`` lets the layout algorithm
        decide; ``"left"`` / ``"right"`` / ``"down"`` / ``"up"`` force a
        specific direction.
    style : str
        Visual style — ``"solid"`` (default), ``"dashed"``, ``"dotted"``.
    """

    field: str
    direction: str = "auto"  # auto | left | right | down | up
    style: str = "solid"  # solid | dashed | dotted


def normalize_edges(raw: list[str | EdgeSpec]) -> list[EdgeSpec]:
    """Accept a mix of plain strings and EdgeSpec objects."""
    out: list[EdgeSpec] = []
    for item in raw:
        if isinstance(item, str):
            out.append(EdgeSpec(field=item))
        elif isinstance(item, EdgeSpec):
            out.append(item)
        else:
            raise TypeError(f"Expected str or EdgeSpec, got {type(item).__name__}")
    return out


# ---------------------------------------------------------------------------
# NodeView — the full visual spec for one class
# ---------------------------------------------------------------------------


@dataclass
class NodeView:
    """Complete visual specification for a registered class.

    Created by :func:`register_class` and embedded in every snapshot node
    so the renderer is self-contained.

    Attributes
    ----------
    shape : str
        Shape to draw — ``"circle"``, ``"box"``, ``"diamond"``.
    label : str
        Field name whose value is rendered as text inside the shape.
    edges : list[EdgeSpec]
        Fields that are structural connections to other nodes.
    data : list[str]
        Extra fields tracked in snapshots but not rendered as edges.
    color_field : str
        Field name whose runtime value determines the node fill color.
        The value is looked up in *color_map* to get the actual color string.
        If empty, the default renderer color is used.
    color_map : dict[str, str]
        Mapping from field value to CSS/manim color string.
        Example: ``{"red": "#CC0000", "black": "#1A1A1A"}``.
    """

    shape: Optional[str] = "circle"
    label: str = ""
    edges: list[EdgeSpec] = field(default_factory=list)
    data: list[str] = field(default_factory=list)
    color_field: str = ""
    color_map: dict[str, str] = field(default_factory=dict)

    # -- derived helpers --------------------------------------------------

    @property
    def fields(self) -> list[str]:
        """All tracked attribute names (label + edge fields + data)."""
        raw: list[str] = []
        if self.label:
            raw.append(self.label)
        raw.extend(e.field for e in self.edges)
        raw.extend(self.data)
        return list(dict.fromkeys(raw))  # deduplicate, preserving order

    @property
    def edge_fields(self) -> set[str]:
        """Set of field names that are edges."""
        return {e.field for e in self.edges}

    def edge_for(self, field_name: str) -> Optional[EdgeSpec]:
        """Return the EdgeSpec for *field_name*, or ``None``."""
        for e in self.edges:
            if e.field == field_name:
                return e
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for embedding in snapshots."""
        d: dict[str, Any] = {
            "shape": self.shape,
            "label": self.label,
            "edges": [
                {"field": e.field, "direction": e.direction, "style": e.style}
                for e in self.edges
            ],
            "data": self.data,
        }
        if self.color_field:
            d["color_field"] = self.color_field
        if self.color_map:
            d["color_map"] = self.color_map
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeView:
        """Reconstruct from a dict (the inverse of :meth:`to_dict`)."""
        return cls(
            shape=d.get("shape", "circle"),
            label=d.get("label", ""),
            edges=[
                EdgeSpec(
                    field=e["field"],
                    direction=e.get("direction", "auto"),
                    style=e.get("style", "solid"),
                )
                for e in d.get("edges", [])
            ],
            data=d.get("data", []),
            color_field=d.get("color_field", ""),
            color_map=d.get("color_map", {}),
        )


# ---------------------------------------------------------------------------
# StyleContext — per-value runtime styling hints
# ---------------------------------------------------------------------------


@dataclass
class StyleContext:
    """
    Lightweight styling hints attached to a specific value at a specific moment.

    Examples::

        StyleContext(highlight={"current": 5})       # highlight node with key 5
        StyleContext(highlight={"visited": [1, 3]})  # mark nodes 1, 3 as visited
        StyleContext(color="red")                     # tint the whole node
    """

    highlight: dict[str, Any] = field(default_factory=dict)
    color: Optional[str] = None
    label: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: StyleContext) -> StyleContext:
        """Return a new StyleContext combining both (other wins on conflict)."""
        return StyleContext(
            highlight={**self.highlight, **other.highlight},
            color=other.color or self.color,
            label=other.label or self.label,
            extras={**self.extras, **other.extras},
        )


class View:
    """
    Base view: decides *how* to render a type structurally.

    Subclass per type (TreeView, ListView, etc.).
    The `render` method receives the value AND an optional StyleContext
    so the same View class can produce different visuals for the same type.
    """

    _type_to_view: dict[type, Callable[[], View]] = {}

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        """
        Produce a renderable representation.
        Subclasses override this to generate manim objects, DOT fragments, etc.
        """
        raise NotImplementedError

    def format_label(self, value: Any) -> str:
        """Format *value* as a short display string (used in node labels).

        Each View subclass overrides this so the formatting is fully
        dispatched through the View system.
        """
        return str(value)

    @staticmethod
    def register(value_type: type, view_factory: Callable[[], View]) -> None:
        """Register a view factory for a specific value type."""
        View._type_to_view[value_type] = view_factory

    @staticmethod
    def for_value(value: Any) -> View:
        """Look up the view for a value's type (walks MRO)."""
        for cls in type(value).__mro__:
            if cls in View._type_to_view:
                return View._type_to_view[cls]()
        raise TypeError(f"No view registered for {type(value).__name__}")

    @classmethod
    def format_value(cls, value: Any) -> str:
        """Format any Python value as display text via View dispatch.

        This is the single entry point used by the renderer to produce
        node-label strings.  Each registered View subclass controls how
        its type is formatted.
        """
        if value is None:
            return NoneView().format_label(value)
        try:
            view = cls.for_value(value)
            return view.format_label(value)
        except TypeError:
            return repr(value)

    @classmethod
    def render_value(cls, value: Any, style: Optional[StyleContext] = None) -> Any:
        """Render any value through the View dispatch system.

        Walks the MRO to find a registered view.  Falls back to a raw
        representation for unregistered types.
        """
        try:
            view = cls.for_value(value)
            return view.render(value, style)
        except TypeError:
            return {"type": "unknown", "value": repr(value), "style": style}


# ---------------------------------------------------------------------------
# Built-in type views
# ---------------------------------------------------------------------------


class StrView(View):
    """View for str — displays the string value."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "str", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)


class BoolView(View):
    """View for bool — displays True/False."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "bool", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)


class NoneView(View):
    """View for None — displays null/empty."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "none", "value": None, "style": style}

    def format_label(self, value: Any) -> str:
        return "\u2205"  # ∅


class IntView(View):
    """View for int — single box with the value."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "int", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)


class FloatView(View):
    """View for float — single box with the value."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "float", "value": value, "style": style}

    def format_label(self, value: Any) -> str:
        return str(value)


class ListView(View):
    """View for list — horizontal/vertical array of cells."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "list", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [View.format_value(item) for item in value]
        return "[" + ", ".join(items) + "]"


class DictView(View):
    """View for dict — key-value table."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = {k: View.render_value(v) for k, v in value.items()}
        return {"type": "dict", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [f"{k}: {View.format_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"


class TupleView(View):
    """View for tuple — similar to list but immutable semantics."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "tuple", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [View.format_value(item) for item in value]
        return "(" + ", ".join(items) + ")"


class SetView(View):
    """View for set — unordered collection of unique items."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "set", "items": items, "style": style}

    def format_label(self, value: Any) -> str:
        items = [View.format_value(item) for item in value]
        return "{" + ", ".join(items) + "}"


class RegisteredClassView(View):
    """View for a ``@register_class``-decorated type.

    Unlike the generic built-in views this one carries the full
    :class:`NodeView` specification (shape, label field, edges, data)
    so that ``View.for_value(obj)`` returns a view that actually knows
    how to describe the node.
    """

    def __init__(self, node_view: NodeView):
        self.node_view = node_view

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {
            "type": "registered_node",
            "node_view": self.node_view.to_dict(),
            "value": value,
            "style": style,
        }

    def format_label(self, value: Any) -> str:
        lf = self.node_view.label
        if lf:
            val = getattr(value, lf, None)
            return View.format_value(val)
        return type(value).__name__


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------

View.register(type(None), lambda: NoneView())
View.register(bool, lambda: BoolView())  # bool before int (bool subclasses int)
View.register(int, lambda: IntView())
View.register(float, lambda: FloatView())
View.register(str, lambda: StrView())
View.register(list, lambda: ListView())
View.register(dict, lambda: DictView())
View.register(tuple, lambda: TupleView())
View.register(set, lambda: SetView())
