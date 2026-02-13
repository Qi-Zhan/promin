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
    """

    shape: str = "circle"
    label: str = ""
    edges: list[EdgeSpec] = field(default_factory=list)
    data: list[str] = field(default_factory=list)

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
        return {
            "shape": self.shape,
            "label": self.label,
            "edges": [
                {"field": e.field, "direction": e.direction, "style": e.style}
                for e in self.edges
            ],
            "data": self.data,
        }

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


class BoolView(View):
    """View for bool — displays True/False."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "bool", "value": value, "style": style}


class NoneView(View):
    """View for None — displays null/empty."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "none", "value": None, "style": style}


class IntView(View):
    """View for int — single box with the value."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "int", "value": value, "style": style}


class FloatView(View):
    """View for float — single box with the value."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "float", "value": value, "style": style}


class ListView(View):
    """View for list — horizontal/vertical array of cells."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "list", "items": items, "style": style}


class DictView(View):
    """View for dict — key-value table."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = {k: View.render_value(v) for k, v in value.items()}
        return {"type": "dict", "items": items, "style": style}


class TupleView(View):
    """View for tuple — similar to list but immutable semantics."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "tuple", "items": items, "style": style}


class SetView(View):
    """View for set — unordered collection of unique items."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        items = [View.render_value(item) for item in value]
        return {"type": "set", "items": items, "style": style}


class TreeView(View):
    """View for tree-like objects — renders as a node graph."""

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "tree", "root": value, "style": style}


class UserView(View):
    """Fully user-defined view with custom shape and label."""

    def __init__(self, shape: str = "box", label: str = ""):
        self.shape = shape
        self.label = label

    def render(self, value: Any, style: Optional[StyleContext] = None) -> Any:
        return {"type": "user", "shape": self.shape, "value": value, "style": style}


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


# ---------------------------------------------------------------------------
# Visual vocabulary constants
# ---------------------------------------------------------------------------

SHAPES: dict[str, str] = {
    "circle": "Circle node (default for trees)",
    "box": "Rectangle box",
    "diamond": "Decision diamond",
}

EDGE_STYLES: dict[str, str] = {
    "solid": "Normal solid line (default)",
    "dashed": "Dashed line",
    "dotted": "Dotted line",
}
