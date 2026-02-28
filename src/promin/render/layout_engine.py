from __future__ import annotations

from typing import Any, Optional

from ..layout import Anchor, LinksLayoutContext, Position, tree
from .snapshot_view import (
    _get_label,
    _get_node_color,
    _get_view,
    _resolve_snapshot,
)
from .types import BOX_HEIGHT, BOX_WIDTH, CONTAINER_PADDING, H_GAP, V_GAP


class _ContainerLayoutContext:
    def __init__(self, gap_x: float = 1.0, gap_y: float = 0.75):
        self.gap_x = gap_x
        self.gap_y = gap_y


class LayoutNode:
    """Intermediate layout element — a container with content."""

    __slots__ = (
        "node_id",
        "label",
        "focused",
        "type_name",
        "shape",
        "fill_color",
        "edge_styles",
        "x",
        "y",
        "children",
        "child_fields",
        "child_kinds",
        "snapshot",
        "type_label",
        "links_layout",
        "content_items",
        "box_width",
        "box_height",
    )

    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.node_id: int | None = snapshot.get("_id")
        self.focused = snapshot.get("_focused", False)
        self.type_name = snapshot.get("_type", "")

        view = _get_view(snapshot)
        self.shape: str | None = view["shape"]
        self.label: str = _get_label(snapshot)
        self.fill_color: Optional[str] = _get_node_color(snapshot)

        self.links_layout = view["links"].get("layout") or tree

        self.type_label: str = ""
        self.content_items = self._build_content_items(view)
        self.box_width, self.box_height = _measure_content_box(self.content_items, self.shape is not None)

        self.child_fields: list[str] = []
        self.children: list[Optional[LayoutNode]] = []
        self.edge_styles: list[dict] = []
        self.child_kinds: list[str] = []

        self._attach_content_children()
        link_object_ids: set[int] = set()
        for link_spec in view["link_specs"]:
            field_name = link_spec["field"]
            if field_name not in snapshot:
                continue
            child = snapshot[field_name]
            if isinstance(child, list):
                for item in child:
                    self.child_fields.append(field_name)
                    edge_spec = dict(link_spec)
                    target = item
                    if isinstance(item, dict) and "target" in item:
                        target = item.get("target")
                        edge_spec["hint"] = item.get("_hint", edge_spec.get("hint", "auto"))
                        edge_spec["style"] = item.get("_style", edge_spec.get("style", "solid"))
                    self.edge_styles.append(edge_spec)
                    if isinstance(target, dict) and "_type" in target:
                        resolved = _resolve_snapshot(target)
                        if resolved is not None and isinstance(resolved, dict) and "_type" in resolved:
                            if resolved.get("_id") is not None:
                                link_object_ids.add(int(resolved["_id"]))
                            self.children.append(LayoutNode(resolved))
                            self.child_kinds.append("link")
                        else:
                            self.children.append(None)
                            self.child_kinds.append("link")
                    else:
                        self.children.append(None)
                        self.child_kinds.append("link")
            elif isinstance(child, dict) and "_type" in child:
                resolved = _resolve_snapshot(child)
                self.child_fields.append(field_name)
                self.edge_styles.append(link_spec)
                if resolved is not None and isinstance(resolved, dict) and "_type" in resolved:
                    if resolved.get("_id") is not None:
                        link_object_ids.add(int(resolved["_id"]))
                    self.children.append(LayoutNode(resolved))
                    self.child_kinds.append("link")
                else:
                    self.children.append(None)
                    self.child_kinds.append("link")
            else:
                self.child_fields.append(field_name)
                self.edge_styles.append(link_spec)
                self.children.append(None)
                self.child_kinds.append("link")
        self._validate_content_link_exclusive(link_object_ids)

    def _build_content_items(self, view: dict) -> list[dict]:
        content_field = view.get("content_field", "")
        raw = self.snapshot.get(content_field, []) if content_field else []
        if not isinstance(raw, list):
            raw = [raw]
        items: list[dict] = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict) and "_type" in item:
                i_view = _get_view(item)
                shape = i_view.get("shape")
                label = _get_label(item)
                fill = _get_node_color(item)
                w, h = _estimate_item_size(label, shape)
                kind = "node_ref" if self.shape is None else "node_body"
                items.append(
                    {
                        "index": idx,
                        "kind": kind,
                        "snapshot_id": item.get("_id"),
                        "snapshot": item,
                        "shape": shape,
                        "text": label,
                        "fill_color": fill,
                        "width": w,
                        "height": h,
                        "dx": 0.0,
                        "dy": 0.0,
                    }
                )
            else:
                text = str(item)
                w, h = _estimate_item_size(text, None)
                items.append(
                    {
                        "index": idx,
                        "kind": "text",
                        "snapshot_id": None,
                        "shape": None,
                        "text": text,
                        "fill_color": None,
                        "width": w,
                        "height": h,
                        "dx": 0.0,
                        "dy": 0.0,
                    }
                )
        _apply_container_layout(items, view["container"].get("layout"))
        return items

    def _attach_content_children(self) -> None:
        for item in self.content_items:
            if item.get("kind") != "node_ref":
                continue
            raw = item.get("snapshot")
            if not isinstance(raw, dict) or "_type" not in raw:
                continue
            resolved = _resolve_snapshot(raw)
            self.child_fields.append("__content")
            self.edge_styles.append(
                {
                    "field": "__content",
                    "style": "none",
                    "hint": "auto",
                    "content_index": int(item.get("index", -1)),
                }
            )
            if resolved is not None and isinstance(resolved, dict) and "_type" in resolved:
                self.children.append(LayoutNode(resolved))
            else:
                self.children.append(None)
            self.child_kinds.append("content")

    def _validate_content_link_exclusive(self, link_object_ids: set[int]) -> None:
        content_ids: set[int] = {
            int(it["snapshot_id"])
            for it in self.content_items
            if it.get("snapshot_id") is not None
        }
        overlap = content_ids & link_object_ids
        if overlap:
            conflict = next(iter(overlap))
            content_idx = next(
                (it["index"] for it in self.content_items if it.get("snapshot_id") == conflict),
                -1,
            )
            link_field = next(
                (
                    f
                    for f, c, k in zip(self.child_fields, self.children, self.child_kinds)
                    if c is not None and c.node_id == conflict and k == "link"
                ),
                "?",
            )
            raise ValueError(
                f"Object id={conflict} appears in both container.show[{content_idx}] and links field '{link_field}'. "
                "A field must belong to either container or links, not both."
            )


def layout_tree(snapshot: Any) -> Optional[LayoutNode]:
    snapshot = _resolve_snapshot(snapshot)
    if snapshot is None or not isinstance(snapshot, dict) or "_type" not in snapshot:
        return None

    root = LayoutNode(snapshot)
    _assign_positions(root, depth=0)
    _resolve_level_overlaps(root)
    _center_tree(root)
    return root


def _assign_positions(node: LayoutNode, depth: int) -> None:
    node.x = 0.0
    node.y = -depth * V_GAP

    def walk(n: LayoutNode, gap_x: float, level: int) -> None:
        content_pairs: list[tuple[LayoutNode, dict]] = [
            (c, spec)
            for c, spec, kind in zip(n.children, n.edge_styles, n.child_kinds)
            if c is not None and kind == "content"
        ]
        link_pairs: list[tuple[LayoutNode, dict]] = [
            (c, spec)
            for c, spec, kind in zip(n.children, n.edge_styles, n.child_kinds)
            if c is not None and kind == "link"
        ]

        for child, spec in content_pairs:
            content_index = spec.get("content_index")
            item = None
            if isinstance(content_index, int):
                item = next((it for it in n.content_items if int(it.get("index", -1)) == content_index), None)
            if item is None:
                continue
            child.x = n.x + float(item.get("dx", 0.0)) / H_GAP
            child.y = n.y + float(item.get("dy", 0.0))
            walk(child, gap_x, level + 1)

        if not link_pairs:
            return

        max_child_w = max(c.box_width for c, _ in link_pairs)
        max_child_h = max(c.box_height for c, _ in link_pairs)
        local_gap_x = max(gap_x, (n.box_width / 2.0 + max_child_w / 2.0 + 0.35) / H_GAP)
        local_gap_y = max(V_GAP, n.box_height / 2.0 + max_child_h / 2.0 + 0.35)

        origin = Anchor(
            id=n.node_id if n.node_id is not None else f"parent-{level}",
            pos=Position(x=n.x, y=n.y),
            meta={"type": n.type_name},
        )
        targets: list[Anchor] = []
        for i, (child, spec) in enumerate(link_pairs):
            targets.append(
                Anchor(
                    id=child.node_id if child.node_id is not None else f"child-{level}-{i}",
                    pos=Position(x=n.x, y=n.y - local_gap_y),
                    meta={
                        "field": spec.get("field"),
                        "hint": spec.get("hint", "auto"),
                        "role": spec.get("role", "child"),
                        "index": i,
                    },
                )
            )

        layout_fn = n.links_layout
        if not callable(layout_fn):
            raise TypeError(f"links.layout for node type '{n.type_name}' must be callable")

        ctx = LinksLayoutContext(
            parent_id=n.node_id,
            gap_x=local_gap_x,
            gap_y=local_gap_y,
            level=level,
            params={},
        )
        laid_out = layout_fn(targets, origin, ctx)
        if not isinstance(laid_out, list):
            raise TypeError("links layout must return list[Anchor]")
        if len(laid_out) != len(targets):
            raise ValueError("links layout must return the same number of anchors as input targets")

        mapping = {a.id: a for a in laid_out}
        for child, anchor_in in zip((c for c, _ in link_pairs), targets):
            anchor = mapping.get(anchor_in.id)
            if anchor is None:
                raise ValueError(f"links layout missing anchor for child id {anchor_in.id}")
            child.x = float(anchor.pos.x)
            child.y = float(anchor.pos.y)
            next_gap_x = local_gap_x
            if getattr(layout_fn, "_promin_layout_kind", "") == "links_tree":
                next_gap_x = max(local_gap_x, local_gap_x * 0.75)
            walk(child, next_gap_x, level + 1)

    walk(node, 1.0, 0)


def _center_tree(root: LayoutNode) -> None:
    xs = list(_collect_x(root))
    if not xs:
        return
    cx = (min(xs) + max(xs)) / 2.0
    _shift_x(root, -cx)


def _collect_x(node: LayoutNode):
    yield node.x
    for c in node.children:
        if c is not None:
            yield from _collect_x(c)


def _shift_x(node: LayoutNode, dx: float):
    node.x += dx
    for c in node.children:
        if c is not None:
            _shift_x(c, dx)


def _flatten_nodes(node: Optional[LayoutNode]) -> list[LayoutNode]:
    if node is None:
        return []
    result = [node]
    for c in node.children:
        result.extend(_flatten_nodes(c))
    return result


def _resolve_level_overlaps(root: LayoutNode) -> None:
    # Push overlapping nodes at the same depth apart by subtree shifts.
    for _ in range(6):
        changed = False
        levels: dict[float, list[LayoutNode]] = {}
        for n in _flatten_nodes(root):
            levels.setdefault(round(float(n.y), 4), []).append(n)
        for _, nodes in levels.items():
            nodes.sort(key=lambda n: n.x)
            for i in range(1, len(nodes)):
                prev = nodes[i - 1]
                cur = nodes[i]
                min_gap_x = (prev.box_width / 2.0 + cur.box_width / 2.0 + 0.2) / H_GAP
                target_x = prev.x + min_gap_x
                if cur.x < target_x:
                    dx = target_x - cur.x
                    _shift_x(cur, dx)
                    changed = True
        if not changed:
            return


def _estimate_item_size(text: str, shape: str | None) -> tuple[float, float]:
    lines = text.splitlines() or [text]
    max_len = max((len(line) for line in lines), default=1)
    if shape is None:
        w = max(0.18, 0.10 * max_len + 0.18)
        h = max(0.20, 0.24 * len(lines) + 0.08)
        return w, h
    w = max(BOX_WIDTH, 0.12 * max_len + 0.35)
    h = max(BOX_HEIGHT, 0.26 * len(lines) + 0.3)
    if shape in ("circle", "diamond"):
        d = max(w, h)
        return d, d
    return w, h


def _apply_container_layout(items: list[dict], layout_fn: Any) -> None:
    if not items:
        return
    targets = [
        Anchor(id=f"content-{i}", pos=Position(0.0, 0.0), meta={"index": i})
        for i in range(len(items))
    ]
    origin = Anchor(id="content-origin", pos=Position(0.0, 0.0))
    max_w = max(float(it["width"]) for it in items)
    max_h = max(float(it["height"]) for it in items)
    ctx = _ContainerLayoutContext(
        gap_x=max(0.75, max_w + 0.2),
        gap_y=max(0.55, max_h + 0.18),
    )
    if layout_fn is None:
        laid_out = [
            t.with_pos(Position(0.0, -(i * ctx.gap_y)))
            for i, t in enumerate(targets)
        ]
    else:
        if not callable(layout_fn):
            raise TypeError("container.layout must be callable")
        laid_out = layout_fn(targets, origin, ctx)
        if not isinstance(laid_out, list) or len(laid_out) != len(targets):
            raise ValueError("container.layout must return list[Anchor] with same length")
    mapping = {a.id: a for a in laid_out}
    for i, item in enumerate(items):
        aid = f"content-{i}"
        if aid in mapping:
            item["dx"] = float(mapping[aid].pos.x)
            item["dy"] = float(mapping[aid].pos.y)
    _normalize_content_offsets(items)


def _normalize_content_offsets(items: list[dict]) -> None:
    """Recenter container content so layout absolute origin does not leak into render."""
    if not items:
        return
    min_x = min(float(it["dx"]) - float(it["width"]) / 2.0 for it in items)
    max_x = max(float(it["dx"]) + float(it["width"]) / 2.0 for it in items)
    min_y = min(float(it["dy"]) - float(it["height"]) / 2.0 for it in items)
    max_y = max(float(it["dy"]) + float(it["height"]) / 2.0 for it in items)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    for it in items:
        it["dx"] = float(it["dx"]) - cx
        it["dy"] = float(it["dy"]) - cy


def _measure_content_box(items: list[dict], include_padding: bool) -> tuple[float, float]:
    if not items:
        return (BOX_WIDTH, BOX_HEIGHT) if include_padding else (0.0, 0.0)
    min_x = min(float(it["dx"]) - float(it["width"]) / 2.0 for it in items)
    max_x = max(float(it["dx"]) + float(it["width"]) / 2.0 for it in items)
    min_y = min(float(it["dy"]) - float(it["height"]) / 2.0 for it in items)
    max_y = max(float(it["dy"]) + float(it["height"]) / 2.0 for it in items)
    w = max_x - min_x
    h = max_y - min_y
    if include_padding:
        w += CONTAINER_PADDING * 2
        h += CONTAINER_PADDING * 2
    return max(w, BOX_WIDTH), max(h, BOX_HEIGHT)
