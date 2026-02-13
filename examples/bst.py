"""
Example: Binary Search Tree

Usage:
    python examples/bst.py
"""

import promin as pm


@pm.register_class(shape="circle", label="key", edges=["left", "right"])
class BSTNode:

    def __init__(self, key: int):
        self.key = key
        self.left = None
        self.right = None

    def insert(self, key: int) -> None:
        if key < self.key:
            if self.left is None:
                self.left = BSTNode(key)
            else:
                self.left.insert(key)
        else:
            if self.right is None:
                self.right = BSTNode(key)
            else:
                self.right.insert(key)

    def search(self, key: int) -> bool:
        if key == self.key:
            return True
        if key < self.key:
            return self.left.search(key) if self.left else False
        return self.right.search(key) if self.right else False

    def keys_inorder(self) -> list[int]:
        result = []
        if self.left:
            result.extend(self.left.keys_inorder())
        result.append(self.key)
        if self.right:
            result.extend(self.right.keys_inorder())
        return result


def bst_build(keys: list[int]) -> BSTNode:
    if not keys:
        raise ValueError("Key list must be non-empty")
    root = BSTNode(keys[0])
    for key in keys[1:]:
        root.insert(key)
    return root


if __name__ == "__main__":
    data = [5, 3, 7, 1, 4, 6, 8]
    root = bst_build(data)
    sm = pm.StateMachine()
    sm.capture(root)
    with pm.record("Search for 4", sm):
        root.search(4)
    sm.render(path="bst_search_4.mp4")

    sm = pm.StateMachine()
    sm.capture(root)
    with pm.record("Insert for 9", sm):
        root.insert(9)
    sm.render(path="bst_insert_9.mp4")
