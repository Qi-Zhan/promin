import pytest

import promin as pm
from promin.render import layout_tree
from promin.tracing.trace import snapshot_objects


def _vertical_layout(ctx: pm.LayoutContext) -> pm.LayoutResult:
    out: dict[int, tuple[float, float]] = {}
    for i, child in enumerate(ctx.children):
        cid = child.get("node_id")
        if cid is None:
            continue
        out[cid] = (0.0, -(i + 1) * 2.0)
    return pm.LayoutResult(out)


@pm.register_type(
    layout=pm.RowLayout(wrap=True, columns=3),
    shape="box",
    label="summary",
    edges=[pm.EdgeSpec(field="elements", direction="right")],
    type_name="list_row_test",
    label_resolver=lambda v: f"len={len(v)}",
    children_resolver=lambda v: {"elements": list(v)},
    data_resolver=lambda v: {"summary": f"len={len(v)}"},
    register_view=False,
)
class _ListRowTest(list):
    pass


def test_builtin_row_layout_wraps():
    snap = snapshot_objects([_ListRowTest([1, 2, 3, 4, 5])])[0]
    root = layout_tree(snap)
    assert root is not None
    children = [c for c in root.children if c is not None]
    assert len(children) == 5
    ys = {round(c.y, 4) for c in children}
    assert len(ys) > 1


def test_custom_layout_callable_applies_coordinates():
    @pm.register_type(
        layout=_vertical_layout,
        shape="circle",
        label="key",
        edges=["left", "right"],
    )
    class _N:
        def __init__(self):
            self.key = 1
            self.left = _Leaf(2)
            self.right = _Leaf(3)

    @pm.register_type(layout=pm.RowLayout(), shape="box", label="v")
    class _Leaf:
        def __init__(self, v: int):
            self.v = v

    snap = snapshot_objects([_N()])[0]
    root = layout_tree(snap)
    assert root is not None
    children = [c for c in root.children if c is not None]
    assert len(children) == 2
    assert children[0].x == pytest.approx(children[1].x)
    assert children[0].y > children[1].y
