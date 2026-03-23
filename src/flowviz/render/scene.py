from __future__ import annotations

from typing import Any

import numpy as np
from manim import DOWN, GREY_A, RIGHT, FadeIn, FadeOut, Line, ReplacementTransform, Scene, Text, VGroup, VMobject

from .geometry import _node_pos
from .layout_engine import LayoutNode, layout_tree
from .mobjects import _make_edge, _make_node_mob
from .types import ANIM_DURATION, RenderConfig, _NodeRenderInfo


def _collect_render_info(root: LayoutNode, origin: np.ndarray) -> dict[int, _NodeRenderInfo]:
    info: dict[int, _NodeRenderInfo] = {}

    def walk(n: LayoutNode) -> None:
        if n.node_id is not None:
            info[n.node_id] = _NodeRenderInfo(
                node_id=n.node_id,
                pos=_node_pos(n, origin),
                shape=n.shape,
                fill_color=n.fill_color,
                focused=n.focused,
                text="",
                width=n.box_width if n.shape is not None else None,
                height=n.box_height if n.shape is not None else None,
                type_label=n.type_label,
                z_index=-2 if n.shape is not None else 0,
                content_items=list(n.content_items),
            )
        for c in n.children:
            if c is not None:
                walk(c)

    walk(root)
    return info


def _collect_layout_edges(root: LayoutNode) -> dict[tuple[int, int], str]:
    edges: dict[tuple[int, int], str] = {}

    def walk(n: LayoutNode) -> None:
        for c, spec in zip(n.children, n.edge_styles):
            if c is not None and n.node_id is not None and c.node_id is not None:
                edges[(n.node_id, c.node_id)] = spec.get("style", "solid")
                walk(c)

    walk(root)
    return edges


class _ManimStateRenderer:
    def __init__(self, scene: Scene, origin: np.ndarray | None = None, config: RenderConfig | None = None):
        self.scene = scene
        self.origin = origin if origin is not None else np.array([0.0, 0.5, 0.0])
        self.config = config or RenderConfig()
        self._node_mobs: dict[int, VGroup] = {}
        self._edge_mobs: dict[tuple[int, int], Line] = {}
        self._overlay: list[VMobject] = []

    def show_state(self, snapshot: Any, loc_text: str = "", counter_text: str = "") -> None:
        payload = build_render_payload(snapshot, origin=self.origin)
        self.show_render_payload(payload, loc_text=loc_text, counter_text=counter_text)

    def show_render_payload(self, payload: dict[str, Any], loc_text: str = "", counter_text: str = "") -> None:
        new_info = _node_info_map_from_payload(payload)
        new_edges = {(int(e["parent"]), int(e["child"])): e["style"] for e in payload.get("edges", [])}

        self._animate_to_state(new_info, new_edges, loc_text=loc_text, counter_text=counter_text)

    def _animate_to_state(
        self,
        new_info: dict[int, _NodeRenderInfo],
        new_edges: dict[tuple[int, int], str],
        *,
        loc_text: str,
        counter_text: str,
    ) -> None:
        anims: list = []

        for m in self._overlay:
            anims.append(FadeOut(m))
        self._overlay.clear()

        if not new_info:
            for m in self._node_mobs.values():
                anims.append(FadeOut(m))
            for m in self._edge_mobs.values():
                anims.append(FadeOut(m))
            self._node_mobs.clear()
            self._edge_mobs.clear()
            if anims:
                self.scene.play(*anims, run_time=ANIM_DURATION * 0.4)
            return

        old_nids = set(self._node_mobs)
        new_nids = set(new_info)
        old_eids = set(self._edge_mobs)
        new_eids = set(new_edges)

        for nid in old_nids - new_nids:
            anims.append(FadeOut(self._node_mobs.pop(nid)))
        for eid in old_eids - new_eids:
            anims.append(FadeOut(self._edge_mobs.pop(eid)))

        for nid in old_nids & new_nids:
            target = _make_node_mob(new_info[nid])
            anims.append(ReplacementTransform(self._node_mobs[nid], target))
            self._node_mobs[nid] = target

        for eid in old_eids & new_eids:
            pid, cid = eid
            style = new_edges[eid]
            p_info, c_info = new_info[pid], new_info[cid]
            new_edge = _make_edge(
                p_info.pos,
                c_info.pos,
                style,
                p_info.shape,
                c_info.shape,
                width1=p_info.width,
                height1=p_info.height,
                width2=c_info.width,
                height2=c_info.height,
            )
            new_edge.set_z_index(-1)
            anims.append(ReplacementTransform(self._edge_mobs[eid], new_edge))
            self._edge_mobs[eid] = new_edge

        for nid in new_nids - old_nids:
            mob = _make_node_mob(new_info[nid])
            self._node_mobs[nid] = mob
            anims.append(FadeIn(mob))
        for eid in new_eids - old_eids:
            pid, cid = eid
            style = new_edges[eid]
            p_info, c_info = new_info[pid], new_info[cid]
            edge = _make_edge(
                p_info.pos,
                c_info.pos,
                style,
                p_info.shape,
                c_info.shape,
                width1=p_info.width,
                height1=p_info.height,
                width2=c_info.width,
                height2=c_info.height,
            )
            edge.set_z_index(-1)
            self._edge_mobs[eid] = edge
            anims.append(FadeIn(edge))

        tc = self.config.text_color
        if isinstance(tc, str) and tc == "auto":
            overlay_color = GREY_A
        else:
            overlay_color = tc or GREY_A
        if loc_text:
            loc_mob = Text(loc_text, font="Menlo", font_size=16, color=overlay_color)
            loc_mob.to_edge(DOWN, buff=0.3)
            self._overlay.append(loc_mob)
            anims.append(FadeIn(loc_mob))

        if counter_text:
            cm = Text(counter_text, font="Menlo", font_size=14, color=overlay_color)
            cm.to_corner(DOWN + RIGHT * 0.1, buff=0.25)
            self._overlay.append(cm)
            anims.append(FadeIn(cm))

        if anims:
            self.scene.play(*anims, run_time=ANIM_DURATION)

    def clear(self) -> None:
        anims: list = []
        for m in self._node_mobs.values():
            anims.append(FadeOut(m))
        for m in self._edge_mobs.values():
            anims.append(FadeOut(m))
        for m in self._overlay:
            anims.append(FadeOut(m))
        if anims:
            self.scene.play(*anims, run_time=ANIM_DURATION * 0.4)
        self._node_mobs.clear()
        self._edge_mobs.clear()
        self._overlay.clear()


def _node_info_to_payload(info: _NodeRenderInfo) -> dict[str, Any]:
    return {
        "node_id": int(info.node_id),
        "pos": [float(info.pos[0]), float(info.pos[1]), float(info.pos[2])],
        "shape": info.shape,
        "fill_color": info.fill_color,
        "focused": bool(info.focused),
        "text": info.text,
        "width": info.width,
        "height": info.height,
        "type_label": info.type_label,
        "z_index": int(info.z_index),
        "content_items": list(info.content_items),
    }


def _node_info_map_from_payload(payload: dict[str, Any]) -> dict[int, _NodeRenderInfo]:
    out: dict[int, _NodeRenderInfo] = {}
    for raw in payload.get("nodes", []):
        pos = raw.get("pos", [0.0, 0.0, 0.0])
        info = _NodeRenderInfo(
            node_id=int(raw["node_id"]),
            pos=np.array([float(pos[0]), float(pos[1]), float(pos[2])]),
            shape=raw["shape"],
            fill_color=raw.get("fill_color"),
            focused=bool(raw.get("focused", False)),
            text=raw.get("text", ""),
            width=raw.get("width"),
            height=raw.get("height"),
            type_label=raw.get("type_label", ""),
            z_index=int(raw.get("z_index", 0)),
            content_items=list(raw.get("content_items", [])),
        )
        out[info.node_id] = info
    return out


def build_render_payload(snapshot: Any, origin: np.ndarray | None = None) -> dict[str, Any]:
    origin_arr = origin if origin is not None else np.array([0.0, 0.5, 0.0])
    snapshots = snapshot if isinstance(snapshot, list) else [snapshot]

    combined_info: dict[int, _NodeRenderInfo] = {}
    combined_edges: dict[tuple[int, int], str] = {}
    for snap in snapshots:
        lr = layout_tree(snap)
        if lr is None:
            continue
        combined_info.update(_collect_render_info(lr, origin_arr))
        combined_edges.update(_collect_layout_edges(lr))

    nodes = [_node_info_to_payload(info) for _, info in sorted(combined_info.items())]
    edges = [
        {"parent": int(pid), "child": int(cid), "style": style}
        for (pid, cid), style in sorted(combined_edges.items())
    ]
    return {"nodes": nodes, "edges": edges}
