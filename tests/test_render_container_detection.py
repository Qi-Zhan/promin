import promin as pm
from promin.render import layout_tree
from promin.tracing.trace import register_type, snapshot_objects


@register_type(
    layout=pm.TreeLayout,
    shape="circle",
    label="key",
    edges=["left", "right"],
)
class _BSTNodeForContainerTest:
    def __init__(self, key: int):
        self.key = key
        self.left = None
        self.right = None


def test_scalar_label_node_is_not_treated_as_container():
    root = _BSTNodeForContainerTest(5)
    root.left = _BSTNodeForContainerTest(3)

    snap = snapshot_objects([root])[0]
    layout = layout_tree(snap)

    assert layout is not None
    assert layout.content_type == "text"
    assert layout.label == "5"
    children = [c for c in layout.children if c is not None]
    assert len(children) == 1
