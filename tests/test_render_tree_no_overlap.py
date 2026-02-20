from promin.render import layout_tree
from promin.trace import snapshot_objects


def test_tree_layout_does_not_overlap_cross_grandchildren():
    class N:
        def __init__(self, key: int):
            self.key = key
            self.left = None
            self.right = None

    import promin as pm

    pm.register_type(
        N,
        layout={"name": "tree", "params": {}},
        shape="circle",
        label="key",
        edges=["left", "right"],
        register_view=False,
    )

    root = N(5)
    root.left = N(3)
    root.right = N(7)
    root.left.right = N(4)
    root.right.left = N(6)

    snap = snapshot_objects([root])[0]
    lr = layout_tree(snap)
    assert lr is not None

    # collect key->(x,y)
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
