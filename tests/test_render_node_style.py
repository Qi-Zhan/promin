import numpy as np

from promin.render.mobjects import _make_node_mob
from promin.render.types import _NodeRenderInfo


def test_colored_circle_node_keeps_solid_fill_style():
    info = _NodeRenderInfo(
        node_id=1,
        pos=np.array([0.0, 0.0, 0.0]),
        shape="circle",
        fill_color="#CC0000",
        focused=False,
        text="",
        width=0.72,
        height=0.72,
        content_items=[{"kind": "text", "text": "7", "width": 0.28, "height": 0.32, "dx": 0.0, "dy": 0.0}],
    )
    mob = _make_node_mob(info)
    body = mob[0]
    assert body.get_fill_opacity() >= 0.8
