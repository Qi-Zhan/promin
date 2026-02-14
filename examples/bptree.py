"""
Example: B+ Tree with insertion & search visualization.

A B+ tree of order *t* stores up to *t-1* keys per node.  When a node
overflows it splits and pushes a separator up to the parent.  All values
live in the leaves; internal nodes only carry separators.

We use order=4 so that splits happen frequently and the tree stays small
enough to visualize.

Usage:
    python examples/bptree.py
"""

from __future__ import annotations

import promin as pm
from promin.trace import State, SourceLoc, compute_transition, snapshot_objects


# ---------------------------------------------------------------------------
# B+ Tree Nodes
# ---------------------------------------------------------------------------

@pm.register_class(
    shape="box",
    label="keys",
    edges=[pm.EdgeSpec(field="children")],
)
class BPInternal:
    """Internal (index) node — keys are separators, children are sub-trees."""

    def __init__(self, keys: list[int] | None = None, children: list | None = None):
        self.keys: list[int] = keys or []
        self.children: list = children or []


@pm.register_class(shape="circle", label="keys")
class BPLeaf:
    """Leaf node — stores sorted data keys."""

    def __init__(self, keys: list[int] | None = None):
        self.keys: list[int] = keys or []


# ---------------------------------------------------------------------------
# B+ Tree operations
# ---------------------------------------------------------------------------

class BPTree:
    """A minimal B+ tree (order *t*) with insert & search."""

    def __init__(self, order: int = 4):
        self.order = order
        self.root: BPInternal | BPLeaf = BPLeaf()

    def insert(self, key: int) -> None:
        result = self._insert_rec(self.root, key)
        if result is not None:
            left, sep, right = result
            self.root = BPInternal(keys=[sep], children=[left, right])

    def search(self, key: int) -> BPLeaf | None:
        """Return the leaf containing *key*, or None."""
        node = self.root
        while isinstance(node, BPInternal):
            i = _child_index(node, key)
            node = node.children[i]
        return node if key in node.keys else None

    def _insert_rec(self, node, key):
        max_keys = self.order - 1
        if isinstance(node, BPLeaf):
            if key not in node.keys:
                node.keys.append(key)
                node.keys.sort()
            if len(node.keys) > max_keys:
                return _split_leaf(node)
            return None
        i = _child_index(node, key)
        result = self._insert_rec(node.children[i], key)
        if result is None:
            return None
        left, sep, right = result
        node.keys.insert(i, sep)
        node.children[i] = left
        node.children.insert(i + 1, right)
        if len(node.keys) > max_keys:
            return _split_internal(node)
        return None


def _child_index(node: BPInternal, key: int) -> int:
    for i, k in enumerate(node.keys):
        if key < k:
            return i
    return len(node.keys)


def _split_leaf(leaf: BPLeaf):
    mid = len(leaf.keys) // 2
    right = BPLeaf(keys=leaf.keys[mid:])
    leaf.keys = leaf.keys[:mid]
    return leaf, right.keys[0], right


def _split_internal(node: BPInternal):
    mid = len(node.keys) // 2
    sep = node.keys[mid]
    right = BPInternal(
        keys=node.keys[mid + 1:],
        children=node.children[mid + 1:],
    )
    node.keys = node.keys[:mid]
    node.children = node.children[:mid + 1]
    return node, sep, right


# ---------------------------------------------------------------------------
# Helper — manually snapshot one state (root can change across inserts)
# ---------------------------------------------------------------------------

def _snap(sm: pm.StateMachine, tree: BPTree, label: str,
          focused_id: int | None = None) -> None:
    """Snapshot the current tree and append a new State to *sm*."""
    sm.captured_objects = [tree.root]
    snapshot = snapshot_objects(sm.captured_objects, focused_id)
    prev = sm.states[-1] if sm.states else None
    sm.states.append(
        State(
            snapshot=snapshot,
            current_loc=SourceLoc("bptree.py", 0, label),
            focused_id=focused_id,
            transition=compute_transition(prev.snapshot, snapshot) if prev else None,
        )
    )


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_insert():
    """Produce a video showing B+ tree insertions."""
    tree = BPTree(order=4)
    sm = pm.StateMachine()

    keys = [10, 20, 5, 6, 12, 30, 7, 17, 25, 3, 8, 15]
    for key in keys:
        tree.insert(key)
        _snap(sm, tree, label=f"insert({key})")

    sm.render(path="media/bptree_insert.mp4", title="B+ Tree Insert (order=4)")


def demo_search():
    """Produce a video showing search traversal."""
    tree = BPTree(order=4)
    for k in [10, 20, 5, 6, 12, 30, 7, 17]:
        tree.insert(k)

    target = 17
    sm = pm.StateMachine()
    _snap(sm, tree, label=f"search({target})")

    # Walk from root to leaf, highlighting each visited node
    node = tree.root
    while isinstance(node, BPInternal):
        i = _child_index(node, target)
        node = node.children[i]
        _snap(sm, tree, label=f"descend → {node.keys}", focused_id=id(node))

    found = target in node.keys
    _snap(sm, tree, label=f"{'found' if found else 'not found'} {target}",
          focused_id=id(node))

    sm.render(path="media/bptree_search.mp4", title=f"B+ Tree Search({target})")


if __name__ == "__main__":
    demo_insert()
    demo_search()
