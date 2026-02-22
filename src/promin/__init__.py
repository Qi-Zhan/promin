__version__ = "0.2.0"

from typing import Callable

from .tracing import trace
from .tracing.trace import (
    register_type,
    override_type_view_spec,
    snapshot_objects,
    State,
    SourceLoc,
    StateMachine,
    Transition,
    NodeChange,
    FieldChange,
    compute_transition,
    record,
)
from .view import (
    EdgeSpec,
    LayoutSpec,
    TypeViewSpec,
    StyleContext,
    View,
    IntView,
    FloatView,
    StrView,
    BoolView,
    NoneView,
    ListView,
    DictView,
    TupleView,
    SetView,
    RegisteredClassView,
)
from .render import (
    register_layout,
    LayoutContext,
    LayoutResult,
    render_states,
    render_states_inline,
    RenderConfig,
)


def register_value_view(value_type: type, view_factory: Callable[[], View]) -> None:
    """Override value rendering and optionally structural type-view behavior.

    Example:
        register_value_view(list, lambda: MyListView())
    """
    View.register(value_type, view_factory)
    view = view_factory()
    spec = view.type_view_spec()
    if spec is not None:
        override_type_view_spec(value_type, spec)


__all__ = [
    "BoolView",
    "DictView",
    "EdgeSpec",
    "FieldChange",
    "FloatView",
    "IntView",
    "LayoutContext",
    "LayoutResult",
    "LayoutSpec",
    "ListView",
    "NodeChange",
    "NoneView",
    "RegisteredClassView",
    "RenderConfig",
    "SetView",
    "SourceLoc",
    "State",
    "StateMachine",
    "StrView",
    "StyleContext",
    "Transition",
    "TupleView",
    "TypeViewSpec",
    "View",
    "compute_transition",
    "override_type_view_spec",
    "record",
    "register_layout",
    "register_type",
    "register_value_view",
    "render_states",
    "render_states_inline",
    "snapshot_objects",
    "trace",
]
