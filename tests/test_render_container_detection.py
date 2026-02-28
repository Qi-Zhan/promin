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
class _BSTNodeForContainerTest:
    def __init__(self, key: int):
        self.key = key
        self.left = None
        self.right = None


def test_scalar_label_node_is_not_treated_as_virtual_container():
    root = _BSTNodeForContainerTest(5)
    root.left = _BSTNodeForContainerTest(3)

    snap = snapshot_objects([root])[0]
    layout = layout_tree(snap)

    assert layout is not None
    assert layout.label == "5"
    children = [c for c in layout.children if c is not None]
    assert len(children) == 1


@(
    pm.type("tree_wrapper")
    .show(lambda t: ["root", t.root])
    .no_view_registration()
)
class _TreeWrapper:
    def __init__(self, root):
        self.root = root


def test_shape_none_container_with_structured_content_expands_subtree():
    root = _BSTNodeForContainerTest(5)
    root.left = _BSTNodeForContainerTest(3)
    root.right = _BSTNodeForContainerTest(7)

    snap = snapshot_objects([_TreeWrapper(root)])[0]
    layout = layout_tree(snap)
    assert layout is not None
    descendants = [c for c in layout.children if c is not None]
    assert descendants
    root_node = descendants[0]
    child_nodes = [c for c in root_node.children if c is not None]
    assert len(child_nodes) == 2
