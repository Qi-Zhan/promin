import pytest

import promin as pm
from promin.render import layout_tree
from promin.trace import snapshot_objects


def test_register_layout_and_use_in_node_layout():
    def zigzag(ctx: pm.LayoutContext) -> pm.LayoutResult:
        out: dict[int, tuple[float, float]] = {}
        for i, child in enumerate(ctx.children):
            cid = child.get("node_id")
            if cid is None:
                continue
            out[cid] = ((-1.0 if i % 2 else 1.0) * (i + 1), -(i + 1) * ctx.gap_y)
        return pm.LayoutResult(positions=out)

    pm.register_layout("zigzag_test", zigzag)

    @pm.register_type(
        layout={"name": "zigzag_test", "params": {}},
        shape="circle",
        label="key",
        edges=["a", "b"],
    )
    class _Node:
        def __init__(self):
            self.key = 1
            self.a = _Leaf(2)
            self.b = _Leaf(3)

    @pm.register_type(
        layout={"name": "row", "params": {}},
        shape="box",
        label="v",
    )
    class _Leaf:
        def __init__(self, v: int):
            self.v = v

    root = _Node()
    snap = snapshot_objects([root])[0]
    lr = layout_tree(snap)
    assert lr is not None
    xs = [c.x for c in lr.children if c is not None]
    assert len(xs) == 2
    assert xs[0] != xs[1]


def test_unknown_layout_name_raises():
    @pm.register_type(
        layout={"name": "does_not_exist_layout", "params": {}},
        shape="circle",
        label="key",
    )
    class _N:
        def __init__(self):
            self.key = 1

    snap = snapshot_objects([_N()])[0]
    with pytest.raises(ValueError, match="Unknown layout"):
        layout_tree(snap)
