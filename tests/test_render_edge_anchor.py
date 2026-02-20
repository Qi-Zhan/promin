import numpy as np

from promin.render import _make_edge


def test_box_vertical_edge_anchors_to_bottom_and_top_faces():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, -1.0, 0.0])
    edge = _make_edge(p1, p2, style="solid", shape1="box", shape2="box")

    start = edge.get_start()
    end = edge.get_end()
    # box half-height is 0.25; start should leave from bottom face, end should enter top face
    assert start[1] < 0.0
    assert end[1] > -1.0


def test_edge_style_none_is_hidden():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    edge = _make_edge(p1, p2, style="none", shape1="box", shape2="box")
    assert edge.get_stroke_width() == 0
