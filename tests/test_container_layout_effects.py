import pytest

import promin as pm
from promin.render import layout_tree
from promin.tracing.trace import snapshot_objects


def _content_row_layout(targets, origin, ctx):
    out = []
    for i, t in enumerate(targets):
        out.append(t.with_pos(pm.Position(origin.pos.x + i * ctx.gap_x, origin.pos.y)))
    return out


@(
    pm.type()
    .shape("box")
    .show(lambda n: [n.a, n.b, n.c])
    .layout(_content_row_layout)
)
class _ContainerRowNode:
    def __init__(self):
        self.a = "A"
        self.b = "B"
        self.c = "C"


@(
    pm.type()
    .shape("box")
    .show(lambda n: n.items)
)
class _BoxNode:
    def __init__(self, items):
        self.items = items


def test_container_layout_changes_content_item_positions():
    snap = snapshot_objects([_ContainerRowNode()])[0]
    root = layout_tree(snap)
    assert root is not None
    xs = [round(float(i["dx"]), 4) for i in root.content_items]
    assert xs == sorted(xs)
    assert xs[-1] > xs[0]
    ys = [round(float(i["dy"]), 4) for i in root.content_items]
    assert len(set(ys)) == 1
    assert abs(ys[0]) < 1e-3


def test_shape_grows_with_more_content_items():
    small = layout_tree(snapshot_objects([_BoxNode(["x"])])[0])
    large = layout_tree(snapshot_objects([_BoxNode(["x", "y", "z", "w"])])[0])
    assert small is not None and large is not None
    assert large.box_height > small.box_height


def test_single_short_content_box_is_tightly_wrapped():
    root = layout_tree(snapshot_objects([_BoxNode(["x"])])[0])
    assert root is not None
    assert root.box_width < 1.2
    assert root.box_height < 1.0


def test_default_container_layout_is_column():
    node = _BoxNode(["a", "b", "c"])
    root = layout_tree(snapshot_objects([node])[0])
    assert root is not None
    ys = [float(i["dy"]) for i in root.content_items]
    assert ys[0] > ys[1] > ys[2]


@(
    pm.type()
    .shape("box")
    .show(lambda n: [n.child])
    .links(
        pm.links()
        .items(lambda n: [n.child])
        .layout(pm.tree)
    )
)
class _ConflictNode:
    def __init__(self):
        self.child = _Leaf("v")


@(
    pm.type()
    .shape("circle")
    .show(lambda n: [n.v])
)
class _Leaf:
    def __init__(self, v):
        self.v = v


def test_container_and_links_overlap_raises_error():
    snap = snapshot_objects([_ConflictNode()])[0]
    with pytest.raises(ValueError, match=r"both container\.show"):
        layout_tree(snap)
