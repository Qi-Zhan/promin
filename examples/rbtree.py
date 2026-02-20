from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import promin as pm


# =====================================================================
# Red-Black Tree data structure
# =====================================================================

RED = "red"
BLACK = "black"
NIL_KEY = "NIL"


@pm.register_class(
    shape="circle",
    label="key",
    edges=[
        pm.EdgeSpec(field="left", direction="left"),
        pm.EdgeSpec(field="right", direction="right"),
    ],
    data=["color"],
    color_field="color",
    color_map={
        "red": "#CC0000",
        "black": "#1A1A1A",
    },
    skip_if=lambda node: node.key == NIL_KEY,
)
@dataclass
class RBNode:
    key: int | str
    color: str = RED
    left: Optional[RBNode] = None
    right: Optional[RBNode] = None
    parent: Optional[RBNode] = None

    @property
    def is_nil(self) -> bool:
        return self.key == NIL_KEY


def _make_nil() -> RBNode:
    n = RBNode(NIL_KEY, BLACK)
    n.left = n.right = n.parent = None  # type: ignore
    return n


SENTINEL = _make_nil()


@pm.register_class(
    shape="box",
    label="root",
)
class RedBlackTree:

    def __init__(self):
        self.nil = SENTINEL
        self.root: RBNode = self.nil

    # -- rotations ------------------------------------------------

    def _left_rotate(self, x: RBNode):
        y = x.right
        x.right = y.left
        if y.left is not self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y: RBNode):
        x = y.left
        y.left = x.right
        if x.right is not self.nil:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y is y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

    # -- insert ---------------------------------------------------

    def insert(self, key: int):
        z = RBNode(key, RED, self.nil, self.nil, None)
        y: Optional[RBNode] = None
        x = self.root
        path: list = []
        while x is not self.nil:
            y = x
            path.append(x.key)
            x = x.left if z.key < x.key else x.right
        z.parent = y
        if y is None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        self._insert_fixup(z)

    def _insert_fixup(self, z: RBNode):
        while z.parent is not None and z.parent.color == RED:
            if z.parent is z.parent.parent.left:  # type: ignore
                uncle = z.parent.parent.right  # type: ignore
                if uncle.color == RED:
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    z.parent.parent.color = RED  # type: ignore
                    z = z.parent.parent  # type: ignore
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED  # type: ignore
                    self._right_rotate(z.parent.parent)  # type: ignore
            else:
                uncle = z.parent.parent.left  # type: ignore
                if uncle.color == RED:
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    z.parent.parent.color = RED  # type: ignore
                    z = z.parent.parent  # type: ignore
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED  # type: ignore
                    self._left_rotate(z.parent.parent)  # type: ignore
        self.root.color = BLACK

    # -- delete ---------------------------------------------------

    def _transplant(self, u: RBNode, v: RBNode):
        """Replace subtree rooted at u with subtree rooted at v."""
        if u.parent is None:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, node: RBNode) -> RBNode:
        while node.left is not self.nil:
            node = node.left
        return node

    def delete(self, key: int):
        z = self._find(self.root, key)
        if z is self.nil:
            return
        y = z
        y_orig_color = y.color
        if z.left is self.nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self.nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_orig_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_orig_color == BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x: RBNode):
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        x.color = BLACK

    def _find(self, node: RBNode, key: int) -> RBNode:
        while node is not self.nil:
            if key == node.key:
                return node
            node = node.left if key < node.key else node.right
        return self.nil

    # -- helpers --------------------------------------------------

    def all_nodes(self) -> list[RBNode]:
        out: list[RBNode] = []
        self._collect(self.root, out)
        return out

    def _collect(self, n: RBNode, out: list[RBNode]):
        if n is self.nil:
            return
        out.append(n)
        self._collect(n.left, out)
        self._collect(n.right, out)


# =====================================================================
# Tests
# =====================================================================


def _rb_inorder(tree: RedBlackTree) -> list[int]:
    out: list[int] = []

    def walk(n: RBNode):
        if n is tree.nil:
            return
        walk(n.left)
        out.append(n.key)  # type: ignore
        walk(n.right)

    walk(tree.root)
    return out


def _rb_valid(tree: RedBlackTree) -> bool:
    """Check RB properties: root black, no red-red, equal black-height."""
    if tree.root.color != BLACK:
        return False

    def check(n: RBNode) -> int:
        if n is tree.nil:
            return 1
        if n.color == RED:
            if (n.left is not tree.nil and n.left.color == RED) or (
                n.right is not tree.nil and n.right.color == RED
            ):
                raise ValueError(f"Red-red violation at {n.key}")
        lh = check(n.left)
        rh = check(n.right)
        if lh != rh:
            raise ValueError(f"Black-height mismatch at {n.key}: {lh} vs {rh}")
        return lh + (1 if n.color == BLACK else 0)

    try:
        check(tree.root)
        return True
    except ValueError:
        return False


def test_rbtree():
    t = RedBlackTree()
    keys = [7, 3, 18, 10, 22, 8, 11, 26, 2, 6, 13]
    for k in keys:
        t.insert(k)
    assert _rb_inorder(t) == sorted(keys)
    assert _rb_valid(t)
    t.delete(18)
    t.delete(11)
    t.delete(3)
    remaining = sorted(set(keys) - {18, 11, 3})
    assert _rb_inorder(t) == remaining
    assert _rb_valid(t)


if __name__ == "__main__":
    test_rbtree()

    # ---- Render insert sequence ----
    t = RedBlackTree()
    keys = [7, 3, 18, 10, 22, 8]

    sm = pm.StateMachine()
    # Capture the whole tree (shape=None wrapper auto-unwraps to root)
    t.insert(keys[0])
    sm.capture(t)

    with pm.record("RBTree insert", sm):
        for k in keys[1:]:
            t.insert(k)

    sm.render(
        path="media/rbtree_insert.gif",
        config=pm.RenderConfig(background_color="#F5F0EB"),
    )
