from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from manim import BLUE_C, GREY_B, YELLOW_C


@dataclass
class RenderConfig:
    """Rendering options passed to ``StateMachine.render()``."""

    background_color: str = ""
    node_color: str = ""
    edge_color: str = ""
    title_color: str = ""
    text_color: str = "auto"
    quality: str = "l"


@dataclass
class _NodeRenderInfo:
    """Unified render info for leaf nodes and containers."""

    node_id: int
    pos: np.ndarray
    shape: str | None
    fill_color: Optional[str] = None
    focused: bool = False
    text: str = ""
    width: Optional[float] = None
    height: Optional[float] = None
    type_label: str = ""
    z_index: int = 0
    content_items: list[dict] = field(default_factory=list)


NODE_RADIUS = 0.30
BOX_WIDTH = 0.60
BOX_HEIGHT = 0.50
H_GAP = 1.3
V_GAP = 1.1
ANIM_DURATION = 0.45

FOCUS_COLOR = YELLOW_C
FOCUS_STROKE = 3.5
NORMAL_FILL = BLUE_C
NORMAL_FILL_OPACITY = 0.22
FOCUS_FILL_OPACITY = 0.45
EDGE_COLOR = GREY_B
EDGE_STROKE = 1.8

CONTAINER_PADDING = 0.20
CONTAINER_STROKE = 1.8
CONTAINER_FILL_OPACITY = 0.06
CONTAINER_LABEL_SIZE = 16
