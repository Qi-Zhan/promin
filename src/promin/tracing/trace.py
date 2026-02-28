"""
promin.tracing.trace — Automatic state tracking via sys.settrace.

Core pipeline::

    1. type_builder(...)(cls)          — declare which types to snapshot
    2. StateMachine.init_state(root)   — capture initial state
    3. record(name, sm)                — trace execution, capture states
    4. sm.render()                     — visualize the state sequence
"""

from __future__ import annotations

import copy
import dis
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from ..render import RenderConfig, render_states, render_tree_text
from .registry import (
    _registered_types,
    type_builder,
)
from ..view import type_view_to_dict

logger = logging.getLogger(__name__)

__all__ = [
    "FieldChange",
    "NodeChange",
    "SourceLoc",
    "State",
    "StateMachine",
    "Transition",
    "compute_transition",
    "record",
    "type_builder",
    "snapshot_objects",
]


@dataclass
class SourceLoc:
    """A source-code location (file, line, function name)."""

    file: str
    line: int
    func: str

    def __repr__(self) -> str:
        return f"{self.func}:{self.line}"


# ---------------------------------------------------------------------------
# Snapshot — deep-serialize a registered object tree into plain dicts
# ---------------------------------------------------------------------------


def _snapshot_values(snapshot: Any, seen: Optional[dict[int, Any]] = None) -> Any:
    """Strip rendering metadata while preserving graph structure in a cycle-safe way."""
    if seen is None:
        seen = {}

    if isinstance(snapshot, (list, tuple)):
        oid = id(snapshot)
        if oid in seen:
            return {"_ref": seen[oid]}
        seen[oid] = f"seq:{len(seen)}"
        return [_snapshot_values(s, seen) for s in snapshot]

    if snapshot is None or not isinstance(snapshot, dict):
        return snapshot

    oid = id(snapshot)
    ref = snapshot.get("_id", f"dict:{len(seen)}")
    if oid in seen:
        return {"_ref": seen[oid]}
    seen[oid] = ref

    if "_type" not in snapshot:
        return {k: _snapshot_values(v, seen) for k, v in snapshot.items()}

    return {
        k: _snapshot_values(v, seen)
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
        "_view": type_view_to_dict(info.view),
    }
    seen[oid] = node  # register before recursion (handles cycles / shared refs)

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
            # Self-references in container/data content should be treated as
            # plain values to avoid creating degenerate self-cycles.
            if data_val is obj:
                node[field_name] = copy.deepcopy(data_val)
            elif field_name == "__content":
                node[field_name] = _snapshot_inline_content(
                    data_val, obj, focused_id, seen
                )
            else:
                node[field_name] = _snapshot_obj_inner(data_val, focused_id, seen)

    for f in info.fields:
        if f in node:
            continue
        node[f] = _snapshot_obj_inner(getattr(obj, f, None), focused_id, seen)
    return node


def _snapshot_inline_content(
    value: Any,
    owner: Any,
    focused_id: Optional[int],
    seen: dict[int, dict],
) -> Any:
    """
    Snapshot container content with plain-list semantics.

    Top-level list/tuple/dict wrappers stay as regular containers instead of
    being interpreted as registered list views. Their element values still
    recurse normally, so registered objects inside content are preserved.
    """
    if value is owner:
        return copy.deepcopy(value)
    if isinstance(value, list):
        return [_snapshot_inline_content(v, owner, focused_id, seen) for v in value]
    if isinstance(value, tuple):
        return [_snapshot_inline_content(v, owner, focused_id, seen) for v in value]
    if isinstance(value, dict):
        return {
            k: _snapshot_inline_content(v, owner, focused_id, seen)
            for k, v in value.items()
        }
    info = _registered_types.get(type(value))
    if info is not None and not info.focusable:
        return copy.deepcopy(value)
    return _snapshot_obj_inner(value, focused_id, seen)


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
    content_field = view.get("container", {}).get("content_field", "")
    if not content_field:
        # fallback for old snapshots
        content_field = view.get("container", {}).get("label", "")
    return node.get(content_field) if content_field else None


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
        # Determine which field is the display content — exclude from diff
        view = new_n.get("_view", {})
        content_field = view.get("container", {}).get("content_field", "")
        if not content_field:
            # fallback for old snapshots
            content_field = view.get("container", {}).get("label", "")
        # Compare all non-meta, non-content fields
        all_fields = {
            k
            for k in list(old_n) + list(new_n)
            if not k.startswith("_") and k != content_field
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

    def _trace(self, frame, event, _arg):
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

    def __exit__(self, *_exc):
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
