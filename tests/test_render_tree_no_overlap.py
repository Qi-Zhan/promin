from promin.render import layout_tree
from promin.render.layout_engine import _flatten_nodes
from promin.render.types import H_GAP
from promin.tracing.trace import snapshot_objects


def test_tree_layout_does_not_overlap_cross_grandchildren():
    class N:
        def __init__(self, key: int):
            self.key = key
            self.left = None
            self.right = None

    import promin as pm

    (
        pm.type()
        .shape("circle")
        .show(lambda n: [n.key])
        .links(
            pm.links()
            .items(lambda n: [n.left, n.right])
            .hints(["left", "right"])
            .layout(pm.tree)
        )
        .no_view_registration()
    )(N)

    root = N(5)
    root.left = N(3)
    root.right = N(7)
    root.left.right = N(4)
    root.right.left = N(6)

    snap = snapshot_objects([root])[0]
    lr = layout_tree(snap)
    assert lr is not None

    coords = {}
    stack = [lr]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        coords[cur.label] = (round(cur.x, 4), round(cur.y, 4))
        for c in cur.children:
            if c is not None:
                stack.append(c)

    assert "4" in coords and "6" in coords
    assert coords["4"] != coords["6"]


def test_tree_layout_keeps_large_container_nodes_non_overlapping():
    class N:
        def __init__(self, idx: int, key: int, tag: str, w: int):
            self.idx = idx
            self.key = key
            self.tag = tag
            self.w = w
            self.left = None
            self.right = None

    import promin as pm

    (
        pm.type()
        .shape("box")
        .show(lambda n: [f"id={n.idx}", f"val={n.key}", f"tag={n.tag}", f"w={n.w}"])
        .layout(pm.row())
        .links(
            pm.links()
            .items(lambda n: [n.left, n.right])
            .hints(["left", "right"])
            .layout(pm.tree)
        )
        .no_view_registration()
    )(N)

    root = N(1, 50, "root", 10)
    root.left = N(2, 30, "L", 7)
    root.right = N(3, 70, "R", 8)
    root.left.left = N(4, 20, "LL", 5)
    root.left.right = N(5, 40, "LR", 6)
    root.right.left = N(6, 60, "RL", 4)
    root.right.right = N(7, 80, "RR", 9)

    snap = snapshot_objects([root])[0]
    lr = layout_tree(snap)
    assert lr is not None

    levels: dict[float, list] = {}
    for n in _flatten_nodes(lr):
        levels.setdefault(round(float(n.y), 4), []).append(n)

    for nodes in levels.values():
        nodes.sort(key=lambda n: n.x)
        for i in range(1, len(nodes)):
            prev = nodes[i - 1]
            cur = nodes[i]
            min_gap_x = (prev.box_width / 2.0 + cur.box_width / 2.0 + 0.2) / H_GAP
            assert cur.x - prev.x >= min_gap_x - 1e-6
