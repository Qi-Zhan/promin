from promin.render import layout_tree
from promin.tracing.trace import snapshot_objects


def test_list_layout_places_elements_left_to_right():
    list_snapshot = snapshot_objects([[1, 2, 3]])[0]
    root = layout_tree(list_snapshot)
    assert root is not None
    children = [c for c in root.children if c is not None]
    assert len(children) == 3
    assert children[0].x < children[1].x < children[2].x


def test_list_layout_wraps_when_too_many_elements():
    list_snapshot = snapshot_objects([list(range(10))])[0]
    root = layout_tree(list_snapshot)
    assert root is not None
    children = [c for c in root.children if c is not None]
    ys = {round(c.y, 4) for c in children}
    # wrap should produce multiple rows (different y values)
    assert len(ys) > 1
