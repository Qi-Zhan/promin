import promin as pm
from promin.render import layout_tree
from promin.tracing.trace import snapshot_objects


@(
    pm.type("list_row_test")
    .shape("box")
    .show(lambda v: [f"len={len(v)}"])
    .links(
        pm.links()
        .items(lambda v: list(v))
        .layout(pm.row(wrap=True, columns=3))
    )
    .no_view_registration()
)
class _ListRowTest(list):
    pass


def test_row_links_layout_wraps_children():
    snap = snapshot_objects([_ListRowTest([1, 2, 3, 4, 5])])[0]
    root = layout_tree(snap)
    assert root is not None
    children = [c for c in root.children if c is not None]
    assert len(children) == 5
    ys = {round(c.y, 4) for c in children}
    assert len(ys) > 1
