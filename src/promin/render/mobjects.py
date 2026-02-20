from __future__ import annotations

import numpy as np

from manim import (
    GREY_A,
    GREY_B,
    UP,
    WHITE,
    Arrow,
    Circle,
    Line,
    ManimColor,
    Polygon,
    Rectangle,
    Text,
    VGroup,
    VMobject,
)

from .geometry import _boundary_offset
from .snapshot_view import _contrast_text_color
from .types import (
    BOX_HEIGHT,
    BOX_WIDTH,
    CONTAINER_FILL_OPACITY,
    CONTAINER_LABEL_SIZE,
    CONTAINER_STROKE,
    EDGE_COLOR,
    EDGE_STROKE,
    FOCUS_COLOR,
    FOCUS_FILL_OPACITY,
    FOCUS_STROKE,
    NODE_RADIUS,
    NORMAL_FILL,
    NORMAL_FILL_OPACITY,
    _NodeRenderInfo,
)


def _make_node_mob(info: _NodeRenderInfo) -> VGroup:
    has_explicit_size = info.width is not None

    if info.fill_color:
        fill = ManimColor(info.fill_color)
        stroke_c = ManimColor(info.fill_color)
        if has_explicit_size:
            opacity = CONTAINER_FILL_OPACITY
            stroke_w = CONTAINER_STROKE
        else:
            opacity = 0.85
            stroke_w = FOCUS_STROKE if info.focused else 2.5
        txt_color = _contrast_text_color(info.fill_color)
        if info.focused and not has_explicit_size:
            stroke_c = FOCUS_COLOR
            stroke_w = FOCUS_STROKE + 1.0
    elif info.focused:
        fill, stroke_c = FOCUS_COLOR, FOCUS_COLOR
        opacity = FOCUS_FILL_OPACITY
        stroke_w = FOCUS_STROKE
        txt_color = WHITE
    else:
        if has_explicit_size:
            fill, stroke_c = GREY_B, GREY_B
            opacity = CONTAINER_FILL_OPACITY
            stroke_w = CONTAINER_STROKE
        else:
            fill, stroke_c = NORMAL_FILL, NORMAL_FILL
            opacity = NORMAL_FILL_OPACITY
            stroke_w = 2.0
        txt_color = WHITE

    pos = info.pos

    if info.shape == "diamond":
        r = max(info.width, info.height) / 2 if has_explicit_size else NODE_RADIUS * 1.2
        body = Polygon(
            pos + np.array([0, r, 0]),
            pos + np.array([r, 0, 0]),
            pos + np.array([0, -r, 0]),
            pos + np.array([-r, 0, 0]),
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )
    elif info.shape == "circle":
        radius = max(info.width, info.height) / 2 if has_explicit_size else NODE_RADIUS
        body = Circle(
            radius=radius,
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )
    else:
        body = Rectangle(
            width=info.width or BOX_WIDTH,
            height=info.height or BOX_HEIGHT,
            color=stroke_c,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=stroke_w,
        )

    body.move_to(pos)
    body.set_z_index(info.z_index)

    parts: list[VMobject] = [body]

    if info.text:
        font_size = 18 if info.shape == "diamond" else 20
        txt = Text(str(info.text), font="Menlo", font_size=font_size, color=txt_color)
        txt.move_to(pos)
        parts.append(txt)

    if info.type_label:
        lbl_color = _contrast_text_color(info.fill_color) if info.fill_color else GREY_A
        lbl = Text(
            info.type_label,
            font="Menlo",
            font_size=CONTAINER_LABEL_SIZE,
            color=lbl_color,
        )
        lbl.move_to(
            body.get_corner(UP + np.array([-1, 0, 0])) + np.array([lbl.width / 2 + 0.12, -0.18, 0])
        )
        lbl.set_z_index(info.z_index)
        parts.append(lbl)

    return VGroup(*parts)


def _make_edge(
    p1: np.ndarray,
    p2: np.ndarray,
    style: str = "solid",
    shape1: str = "circle",
    shape2: str = "circle",
) -> Line:
    d = p2 - p1
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-6:
        return Line(p1, p2, color=EDGE_COLOR, stroke_width=0)
    dn = d / d_len
    start = p1 + _boundary_offset(shape1, dn)
    end = p2 - _boundary_offset(shape2, dn)

    if style == "none":
        return Line(start, end, color=EDGE_COLOR, stroke_width=0)

    edge = Arrow(
        start,
        end,
        buff=0,
        stroke_width=EDGE_STROKE,
        color=EDGE_COLOR,
        max_tip_length_to_length_ratio=0.15,
        max_stroke_width_to_length_ratio=8,
    )
    if style == "dashed":
        edge.set_stroke(opacity=0.7)
    elif style == "dotted":
        edge.set_stroke(opacity=0.5)
    return edge
