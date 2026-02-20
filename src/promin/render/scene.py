from __future__ import annotations

from typing import Any

import numpy as np
from manim import DOWN, GREY_A, RIGHT, FadeIn, FadeOut, Line, ReplacementTransform, Scene, Text, VGroup, VMobject

from .geometry import _compute_bounding_box, _node_pos
from .layout_engine import LayoutNode, layout_tree
from .mobjects import _make_edge, _make_node_mob
from .types import ANIM_DURATION, CONTAINER_PADDING, RenderConfig, _NodeRenderInfo


def _collect_render_info(root: LayoutNode, origin: np.ndarray) -> dict[int, _NodeRenderInfo]:
    info: dict[int, _NodeRenderInfo] = {}

    def walk(n: LayoutNode) -> None:
        if n.node_id is not None:
            if n.content_type == "subtree" and n.content_root is not None:
                x_min, x_max, y_min, y_max = _compute_bounding_box(n.content_root, origin)
                pad = CONTAINER_PADDING
                info[n.node_id] = _NodeRenderInfo(
                    node_id=n.node_id,
                    pos=np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0]),
                    shape=n.shape,
                    fill_color=n.fill_color,
                    focused=n.focused,
                    width=max(x_max - x_min + pad * 2, 1.2),
                    height=max(y_max - y_min + pad * 2, 0.9),
                    type_label=n.type_label,
                    z_index=-2,
                )
                walk(n.content_root)
            else:
                info[n.node_id] = _NodeRenderInfo(
                    node_id=n.node_id,
                    pos=_node_pos(n, origin),
                    shape=n.shape,
                    fill_color=n.fill_color,
                    focused=n.focused,
                    text=n.label,
                )
        for c in n.children:
            if c is not None:
                walk(c)

    walk(root)
    return info


def _collect_layout_edges(root: LayoutNode) -> dict[tuple[int, int], str]:
    edges: dict[tuple[int, int], str] = {}

    def walk(n: LayoutNode) -> None:
        if n.content_type == "subtree" and n.content_root is not None:
            walk(n.content_root)
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
        snapshots = snapshot if isinstance(snapshot, list) else [snapshot]

        new_info: dict[int, _NodeRenderInfo] = {}
        new_edges: dict[tuple[int, int], str] = {}
        for snap in snapshots:
            lr = layout_tree(snap)
            if lr is not None:
                new_info.update(_collect_render_info(lr, self.origin))
                new_edges.update(_collect_layout_edges(lr))

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
            new_edge = _make_edge(p_info.pos, c_info.pos, style, p_info.shape, c_info.shape)
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
            edge = _make_edge(p_info.pos, c_info.pos, style, p_info.shape, c_info.shape)
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
