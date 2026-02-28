import promin as pm
from promin.render import layout_tree
from promin.tracing.trace import snapshot_objects


@(
    pm.type()
    .shape("circle")
    .show(lambda n: [n.key])
    .links(
        pm.links()
        .items(lambda n: [n.left, n.right])
        .hints(["left", "right"])
        .layout(pm.tree)
    )
)
class _TreeNode:
    def __init__(self, key: int):
        self.key = key
        self.left = None
        self.right = None


def test_tree_layout_uses_link_hints_for_left_right_split():
    root = _TreeNode(1)
    root.left = _TreeNode(2)
    root.right = _TreeNode(3)

    snap = snapshot_objects([root])[0]
    lr = layout_tree(snap)
    assert lr is not None
    children = [c for c in lr.children if c is not None]
    assert len(children) == 2

    by_label = {c.label: c for c in children}
    assert by_label["2"].x < 0.0
    assert by_label["3"].x > 0.0


def test_custom_links_position_layout_function_applies_coordinates():
    def custom_links_layout(targets, origin, ctx):
        out = []
        for i, t in enumerate(targets):
            out.append(t.with_pos(pm.Position(origin.pos.x, origin.pos.y - (i + 1) * 2.0)))
        return out

    @(
        pm.type()
        .shape("circle")
        .show(lambda n: [n.key])
        .links(
            pm.links()
            .items(lambda n: [n.a, n.b])
            .layout(custom_links_layout)
        )
    )
    class _N:
        def __init__(self):
            self.key = 1
            self.a = _TreeNode(10)
            self.b = _TreeNode(20)

    snap = snapshot_objects([_N()])[0]
    root = layout_tree(snap)
    assert root is not None
    children = [c for c in root.children if c is not None]
    assert len(children) == 2
    assert children[0].x == children[1].x
    assert children[0].y > children[1].y
