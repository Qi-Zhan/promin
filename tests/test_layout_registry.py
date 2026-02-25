import pytest

import promin as pm
from promin.render import layout_tree
from promin.tracing.trace import snapshot_objects


def test_custom_layout_callable_and_use_in_node_layout():
    def zigzag(ctx: pm.LayoutContext) -> pm.LayoutResult:
        out: dict[int, tuple[float, float]] = {}
        for i, child in enumerate(ctx.children):
            cid = child.get("node_id")
            if cid is None:
                continue
            out[cid] = ((-1.0 if i % 2 else 1.0) * (i + 1), -(i + 1) * ctx.gap_y)
        return pm.LayoutResult(positions=out)

    @pm.register_type(
        layout=zigzag,
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
        layout=pm.RowLayout(),
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


def test_unknown_builtin_layout_kind_raises():
    @pm.register_type(
        layout=pm.TreeLayout,
        shape="circle",
        label="key",
        edges=["left"],
    )
    class _N:
        def __init__(self):
            self.key = 1
            self.left = None

    n = _N()
    n.left = _N()
    snap = snapshot_objects([n])[0]
    snap["_view"]["layout"] = object()
    with pytest.raises(TypeError, match="must be callable"):
        layout_tree(snap)


def test_register_layout_removed_from_public_api():
    assert not hasattr(pm, "register_layout")


def test_register_type_rejects_dict_layout():
    @pm.register_type(
        layout=pm.TreeLayout,
        shape="circle",
        label="key",
    )
    class _Tmp:
        def __init__(self):
            self.key = 1

    with pytest.raises(TypeError, match="must be callable"):
        pm.register_type(layout={"name": "tree", "params": {}}, shape="circle", label="key")(
            _Tmp
        )
