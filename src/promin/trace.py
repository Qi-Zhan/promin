"""
promin.trace — Automatic state tracking via sys.settrace.

Core pipeline::

    1. register_type(...)              — declare which types to snapshot
    2. StateMachine.init_state(root)   — capture initial state
    3. record(name, sm)                — trace execution, capture states
    4. sm.render()                     — visualize the state sequence
"""

from __future__ import annotations

import copy
import json
import logging
import dis
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .view import (
    TypeViewSpec,
    EdgeSpec,
    LayoutSpec,
    normalize_edges,
    normalize_layout_spec,
    View,
    RegisteredClassView,
)
from .render import render_tree_text, render_states, RenderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives — lightweight data carriers used throughout the module
# ---------------------------------------------------------------------------


@dataclass
class SourceLoc:
    """A source-code location (file, line, function name)."""

    file: str
    line: int
    func: str

    def __repr__(self) -> str:
        return f"{self.func}:{self.line}"


@dataclass
class _TypeInfo:
    """Internal metadata for a registered type."""

    type_name: str
    view: TypeViewSpec
    typ: type
    skip_if: Optional[Callable] = None
    label_resolver: Optional[Callable[[Any], Any]] = None
    children_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    data_resolver: Optional[Callable[[Any], dict[str, Any]]] = None
    focusable: bool = True

    @property
    def fields(self) -> list[str]:
        return self.view.fields


# ---------------------------------------------------------------------------
# Registry — which classes are tracked and how to snapshot them
# ---------------------------------------------------------------------------

_registered_types: dict[type, _TypeInfo] = {}


def override_type_view_spec(value_type: type, view_spec: TypeViewSpec) -> bool:
    """Override the registered TypeViewSpec for an existing type.

    Returns True when the type exists in the snapshot registry, False otherwise.
    """
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
    """
    Register a type for automatic state tracking and visualization.

    Example::

        @pm.register_type(
            layout={"name": "tree", "params": {}},
            shape="circle",
            label="key",
            edges=["left", "right"],
        )
        class BSTNode:
            ...

    Parameters
    ----------
    shape : str
        Visual shape — ``"circle"`` (default), ``"box"``, ``"diamond"``.
    label : str
        Field name whose value is rendered as text inside the shape.
    edges : list[str | EdgeSpec]
        Fields that represent connections to other nodes.  Plain strings
        get ``direction="auto"``; use :class:`EdgeSpec` for finer control.
    data : list[str]
        Extra fields to track in snapshots (not rendered as connections).
    type_name : str
        Logical name for rendering (defaults to ``cls.__name__``).
    color_field : str
        Field name whose runtime value determines the node fill color.
    color_map : dict[str, str]
        Mapping from *color_field* values to actual color strings.
    skip_if : callable | None
        Predicate ``(obj) -> bool``.  When it returns ``True`` the object
        is treated as ``None`` during snapshotting (useful for sentinel
        nodes in Red-Black trees, etc.).
    """

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
            typ=target_cls,
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
        # User-defined object types participate in View dispatch as registered nodes.
        # Built-ins can keep their dedicated View implementations by setting
        # register_view=False.
        if register_view:
            View.register(target_cls, lambda v=view: RegisteredClassView(v))
        return target_cls

    if cls is not None:
        return _register(cls)

    def decorator(target_cls):
        return _register(target_cls)

    return decorator


def _register_builtin_types() -> None:
    """Register phase-1 built-in types in the unified registry."""
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


# ---------------------------------------------------------------------------
# Snapshot — deep-serialize a registered object tree into plain dicts
# ---------------------------------------------------------------------------


def _snapshot_values(snapshot: Any) -> Any:
    """Strip ``_id``, ``_focused``, and ``_view`` from a snapshot, keeping only value content."""
    if isinstance(snapshot, (list, tuple)):
        return [_snapshot_values(s) for s in snapshot]
    if snapshot is None or not isinstance(snapshot, dict) or "_type" not in snapshot:
        return snapshot
    return {
        k: _snapshot_values(v)
        for k, v in snapshot.items()
        if k not in ("_id", "_focused", "_view")
    }


def _stable_child_id(parent_oid: int, field_name: str, index: int) -> int:
    """Stable synthetic id for container child slots."""
    return hash((parent_oid, field_name, index))


def _snapshot_obj_inner(
    obj: Any, focused_id: Optional[int], seen: dict[int, dict], *,
    synthetic_id: Optional[int] = None,
) -> Any:
    """Recursive snapshot helper with a shared *seen* dict for dedup."""
    if obj is None:
        return None

    info = _registered_types.get(type(obj))
    if info is None:
        # Non-phase-1 builtins keep current fallback semantics.
        if isinstance(obj, tuple):
            return tuple(_snapshot_obj_inner(item, focused_id, seen) for item in obj)
        if isinstance(obj, set):
            return [_snapshot_obj_inner(item, focused_id, seen) for item in obj]
        if isinstance(obj, dict):
            return {k: _snapshot_obj_inner(v, focused_id, seen) for k, v in obj.items()}
        return copy.deepcopy(obj)

    # skip_if predicate — treat excluded objects as None (e.g. sentinel nodes)
    if info.skip_if is not None and info.skip_if(obj):
        return None

    oid = synthetic_id if synthetic_id is not None else id(obj)
    if oid in seen:
        return seen[oid]

    node: dict[str, Any] = {
        "_type": info.type_name,
        "_id": oid,
        "_focused": (oid == focused_id) if focused_id else False,
        "_view": info.view.to_dict(),
    }
    seen[oid] = node  # register before recursion (handles cycles / shared refs)
    if info.label_resolver is not None:
        label_field = info.view.label
        if label_field:
            node[label_field] = copy.deepcopy(info.label_resolver(obj))

    if info.children_resolver is not None:
        for field_name, child_val in info.children_resolver(obj).items():
            if isinstance(child_val, list):
                children = []
                for i, item in enumerate(child_val):
                    children.append(
                        _snapshot_obj_inner(
                            item,
                            focused_id,
                            seen,
                            synthetic_id=_stable_child_id(oid, field_name, i),
                        )
                    )
                node[field_name] = children
            else:
                node[field_name] = _snapshot_obj_inner(child_val, focused_id, seen)

    if info.data_resolver is not None:
        for field_name, data_val in info.data_resolver(obj).items():
            node[field_name] = copy.deepcopy(data_val)

    for f in info.fields:
        if f in node:
            continue
        node[f] = _snapshot_obj_inner(getattr(obj, f, None), focused_id, seen)
    return node


def snapshot_objects(objs: list[Any], focused_id: Optional[int] = None) -> list[Any]:
    """
    Snapshot multiple registered-class objects in a single pass.

    Unlike calling :func:`snapshot_object` per root, this function uses a
    shared ``seen`` set so that objects referenced by more than one root
    appear exactly once (deduplicated by ``id()``).
    """
    seen: dict[int, dict] = {}
    return [_snapshot_obj_inner(obj, focused_id, seen) for obj in objs]


# ---------------------------------------------------------------------------
# Transition & State — the data captured at each step
# ---------------------------------------------------------------------------


@dataclass
class NodeChange:
    """A node that was added or removed."""

    node_id: int
    key: Any = None
    type_name: str = ""

    def __repr__(self) -> str:
        return f"{self.type_name}(key={self.key}, id={self.node_id})"


@dataclass
class FieldChange:
    """A field-level value change on a persisted node."""

    node_id: int
    field: str
    old_value: Any = None
    new_value: Any = None
    key: Any = None
    type_name: str = ""

    def __repr__(self) -> str:
        return (
            f"{self.type_name}(key={self.key}).{self.field}: "
            f"{self.old_value!r} -> {self.new_value!r}"
        )


@dataclass
class Transition:
    """Detailed diff between two consecutive states.

    Attributes
    ----------
    added : list[NodeChange]
        Nodes that appeared since the previous state.
    removed : list[NodeChange]
        Nodes that disappeared.
    modified : list[FieldChange]
        Per-field value changes within nodes that persist across states.
    old_focus_id : int | None
        Previously focused node (``None`` if no focus).
    new_focus_id : int | None
        Currently focused node.
    """

    added: list[NodeChange]
    removed: list[NodeChange]
    modified: list[FieldChange]
    old_focus_id: Optional[int]
    new_focus_id: Optional[int]


def _iter_snapshot_nodes(snapshot: Any):
    """Yield every registered-object node (dict with ``_id``) in *snapshot*.

    Handles lists, tuples, and nested dicts recursively.
    """
    if isinstance(snapshot, (list, tuple)):
        for item in snapshot:
            yield from _iter_snapshot_nodes(item)
        return
    if not isinstance(snapshot, dict) or "_id" not in snapshot:
        return
    yield snapshot
    for v in snapshot.values():
        if isinstance(v, (dict, list, tuple)):
            yield from _iter_snapshot_nodes(v)


def _node_label(node: dict) -> Any:
    """Extract the display label value from a snapshot node using its ``_view``."""
    view = node.get("_view", {})
    label_field = view.get("label", "")
    return node.get(label_field) if label_field else None


# --- snapshot walkers (all built on _iter_snapshot_nodes) -----------------


def _collect_nodes(snapshot: Any) -> dict[int, dict]:
    """Collect all nodes from a snapshot (or list of snapshots) keyed by ``_id``."""
    return {n["_id"]: n for n in _iter_snapshot_nodes(snapshot)}


def _find_focus_id(snapshot: Any) -> Optional[int]:
    """Return the ``_id`` of the focused node, or ``None``."""
    return next(
        (n["_id"] for n in _iter_snapshot_nodes(snapshot) if n.get("_focused")),
        None,
    )


def _find_label_by_id(snapshot: Any, target_id: int) -> Any:
    """Return the display label of the node matching *target_id*."""
    for n in _iter_snapshot_nodes(snapshot):
        if n["_id"] == target_id:
            return _node_label(n)
    return None


def _summarize_value(val: Any) -> Any:
    """Return a display-friendly summary of a snapshot value."""
    if val is None:
        return None
    if isinstance(val, dict) and "_type" in val:
        label = _node_label(val)
        return f'{val["_type"]}({label!r})' if label is not None else val["_type"]
    return val


def compute_transition(old_snap: Any, new_snap: Any) -> Transition:
    """Compute a detailed structural + value diff between two snapshots."""
    old_nodes = _collect_nodes(old_snap) if old_snap else {}
    new_nodes = _collect_nodes(new_snap) if new_snap else {}

    old_ids = set(old_nodes)
    new_ids = set(new_nodes)

    added = []
    for nid in sorted(new_ids - old_ids):
        n = new_nodes[nid]
        added.append(
            NodeChange(node_id=nid, key=_node_label(n), type_name=n.get("_type", ""))
        )

    removed = []
    for nid in sorted(old_ids - new_ids):
        n = old_nodes[nid]
        removed.append(
            NodeChange(node_id=nid, key=_node_label(n), type_name=n.get("_type", ""))
        )

    modified = []
    for nid in sorted(old_ids & new_ids):
        old_n = old_nodes[nid]
        new_n = new_nodes[nid]
        type_name = new_n.get("_type", "")
        label = _node_label(new_n)
        # Determine which field is the label (identity) — exclude from diff
        view = new_n.get("_view", {})
        label_field = view.get("label", "")
        # Compare all non-meta, non-label fields
        all_fields = {
            k
            for k in list(old_n) + list(new_n)
            if not k.startswith("_") and k != label_field
        }
        for field in sorted(all_fields):
            old_val = _snapshot_values(old_n.get(field))
            new_val = _snapshot_values(new_n.get(field))
            if old_val != new_val:
                modified.append(
                    FieldChange(
                        node_id=nid,
                        field=field,
                        old_value=_summarize_value(old_n.get(field)),
                        new_value=_summarize_value(new_n.get(field)),
                        key=label,
                        type_name=type_name,
                    )
                )

    transition = Transition(
        added=added,
        removed=removed,
        modified=modified,
        old_focus_id=_find_focus_id(old_snap),
        new_focus_id=_find_focus_id(new_snap),
    )
    return transition


# _find_key_by_id replaced by _find_label_by_id (defined above)


@dataclass
class State:
    """
    A snapshot of the tracked object tree at one moment in time.

    Attributes
    ----------
    snapshot : dict | Any
        Dict tree produced by :func:`snapshot_object`.
    current_loc : SourceLoc | None
        Source location that produced this state.
    focused_id : int | None
        ``id()`` of the currently-active object (highlighted in views).
    transition : Transition | None
        What changed from the previous state (``None`` for the initial state).
    """

    snapshot: Any
    current_loc: Optional[SourceLoc] = None
    focused_id: Optional[int] = None
    transition: Optional[Transition] = None

    @classmethod
    def init(cls, roots: list[Any]) -> State:
        """Create an initial state from a root object (no location, no focus)."""
        return cls(snapshot=snapshot_objects(roots))

    def __repr__(self) -> str:
        loc = f" @ {self.current_loc}" if self.current_loc else " (initial)"
        focus = ""
        if self.focused_id:
            snap = self.snapshot if isinstance(self.snapshot, list) else [self.snapshot]
            for s in snap:
                label = _find_label_by_id(s, self.focused_id)
                if label is not None:
                    focus = f" [focus: {label}]"
                    break
        return f"State{loc}{focus}"


# ---------------------------------------------------------------------------
# StateMachine — ordered sequence of States
# ---------------------------------------------------------------------------


class StateMachine:
    """
    Holds the full sequence of states captured during a :func:`record` session.
    """

    def __init__(self):
        self.states: list[State] = []
        self.captured_objects: list[Any] = []

    def capture(self, root: Any) -> None:
        self.captured_objects.append(root)

    def init(self):
        assert self.captured_objects
        assert len(self.states) == 0, "StateMachine already initialized"
        initial_state = State.init(self.captured_objects)
        self.states.append(initial_state)
        logger.info("StateMachine.init: initial state captured")

    def render(
        self,
        path: str = "",
        fps: int = 30,
        title: str = "",
        config: RenderConfig | None = None,
    ) -> None:
        """
        Render the recorded states.

        If *path* ends with a video extension (``.mp4``, ``.mov``, ``.webm``,
        or ``.gif``), render media using that suffix.

        Parameters
        ----------
        path : str
            Output file path. Empty or non-video → text dump.
        fps : int
            Frames per second (video only).
        title : str
            Title shown at the top of the video.
        config : RenderConfig | None
            Rendering options (background color, quality, etc.).
        """
        video_exts = {".mp4", ".mov", ".webm", ".gif"}
        if path and any(path.endswith(ext) for ext in video_exts):
            logger.info(
                "StateMachine.render: video path=%s fps=%d title=%s",
                path,
                fps,
                title,
            )
            out = render_states(
                self.states, path, fps=fps, title=title, config=config
            )
            logger.info("Rendered %d states -> %s", len(self.states), out)
            return

        # Fallback: text visualization
        bar = "═" * 60
        logger.info("\n%s", bar)
        logger.info("StateMachine: %d states", len(self.states))
        logger.info("StateMachine.render: text mode states=%d", len(self.states))
        logger.info("%s\n", bar)

        for i, state in enumerate(self.states):
            loc_str = f"@ {state.current_loc}" if state.current_loc else "(initial)"
            logger.info("S%d  %s", i, loc_str)
            logger.info("\n%s", render_tree_text(state.snapshot, indent=6))

    def __repr__(self) -> str:
        return f"StateMachine({len(self.states)} states)"


# ---------------------------------------------------------------------------
# record() — context manager powered by sys.settrace
# ---------------------------------------------------------------------------

_LOAD_OPS = frozenset(
    {
        "LOAD_FAST",
        "LOAD_FAST_CHECK",
        "LOAD_NAME",
        "LOAD_DEREF",
        "LOAD_ATTR",
    }
)


def _is_focusable_registered_value(val: Any) -> bool:
    info = _registered_types.get(type(val))
    return info is not None and info.focusable


def _line_var_names(code, lineno: int) -> list[str]:
    """Return variable names referenced by bytecode instructions on *lineno*."""
    names: list[str] = []
    for instr in dis.get_instructions(code):
        if instr.positions is not None:
            if instr.positions.lineno != lineno:
                continue
        elif getattr(instr, "starts_line", None) != lineno:
            continue
        if instr.opname in _LOAD_OPS and instr.argval is not None:
            names.append(instr.argval)
    return names


class _RecordContext:
    """
    Context manager that uses ``sys.settrace`` to capture a :class:`State`
    every time a registered-class method executes a source line.
    """

    def __init__(self, scope_name: str, sm: StateMachine, trace_current: bool = True):
        self.scope_name = scope_name
        self.sm = sm
        self.sm.init()
        self._prev_trace = None
        self._base_dir = os.getcwd()
        self._lib_dir = os.path.dirname(os.path.abspath(__file__))
        self._trace_current = trace_current
        logger.info(
            "record: scope=%s trace_current=%s", scope_name, trace_current
        )

    # ---- helpers ---------------------------------------------------------

    def _emit_state(self, frame, focused_id: Optional[int], new_data: list) -> None:
        """Build and append a new State from the current frame context."""
        snapshot = (
            snapshot_objects(self.sm.captured_objects, focused_id)
            if focused_id
            else new_data
        )
        prev = self.sm.states[-1]
        self.sm.states.append(
            State(
                snapshot=snapshot,
                current_loc=SourceLoc(
                    file=os.path.relpath(frame.f_code.co_filename, self._base_dir),
                    line=frame.f_lineno,
                    func=frame.f_code.co_name,
                ),
                focused_id=focused_id,
                transition=compute_transition(prev.snapshot, snapshot),
            )
        )

    # ---- trace callback --------------------------------------------------

    def _trace(self, frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if not filename.startswith(self._base_dir):
                return None
            if filename.startswith(self._lib_dir):
                return None
            return self._trace

        if event == "return":
            self._capture_on_return(frame)
            return self._trace

        if event != "line":
            return self._trace

        # Only consider variables referenced by the current line's bytecode
        names = _line_var_names(frame.f_code, frame.f_lineno)
        focused_obj = None
        for name in names:
            val = frame.f_locals.get(name)
            if val is not None and _is_focusable_registered_value(val):
                focused_obj = val
                break

        if focused_obj is None:
            return self._trace

        prev = self.sm.states[-1]
        focused_id = id(focused_obj) if self._trace_current else None

        # Case 1: data values changed (insertion, deletion, mutation …)
        new_data = snapshot_objects(self.sm.captured_objects)
        values_changed = _snapshot_values(new_data) != _snapshot_values(prev.snapshot)

        # Case 2: focus moved to a different node (only when trace_current)
        focus_changed = self._trace_current and (focused_id != prev.focused_id)

        if not values_changed and not focus_changed:
            return self._trace

        self._emit_state(frame, focused_id, new_data)
        return self._trace

    # ---- return-event capture --------------------------------------------

    def _capture_on_return(self, frame) -> None:
        """Capture state on function return if the last line mutated data."""
        prev = self.sm.states[-1]
        new_data = snapshot_objects(self.sm.captured_objects)
        if _snapshot_values(new_data) == _snapshot_values(prev.snapshot):
            return

        # Try to find a focused object from locals
        focused_obj = None
        if self._trace_current:
            for val in frame.f_locals.values():
                if val is not None and _is_focusable_registered_value(val):
                    focused_obj = val
                    break

        focused_id = id(focused_obj) if focused_obj else None
        self._emit_state(frame, focused_id, new_data)

    # ---- context protocol ------------------------------------------------

    def __enter__(self):
        self._prev_trace = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._prev_trace)
        return False


def record(
    scope_name: str, state_machine: StateMachine, trace_current: bool = True
) -> _RecordContext:
    """
    Context manager: trace execution and capture states automatically.

    Example::

        sm = pm.StateMachine.init_state(root)
        with pm.record("Search for 4", sm):
            root.search(4)
        sm.render(path="output.mp4")

    Parameters
    ----------
    scope_name : str
        Label for this recording session.
    state_machine : StateMachine
        Where captured states are appended.
    trace_current : bool
        Whether to include the currently-focused object in snapshots (default: True).
    """
    return _RecordContext(scope_name, state_machine, trace_current=trace_current)
