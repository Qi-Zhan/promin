"""
Example: container layout + links layout together.

Usage:
    uv run python examples/container_and_links_layout.py
"""

import promin as pm


@(
    pm.type()
    .shape("box")
    .show(lambda n: [n.id, n.val, n.tag, n.weight])
    .layout(pm.grid(columns=2))
    .links(
        pm.links()
        .items(lambda n: [n.left, n.right])
        .hints(["left", "right"])
        .layout(pm.tree)
    )
)
class HybridNode:
    def __init__(self, id: int, val: int, tag: str, weight: int):
        self.id = id
        self.val = val
        self.tag = tag
        self.weight = weight
        self.left = None
        self.right = None

    def insert(self, node: "HybridNode") -> None:
        if node.val < self.val:
            if self.left is None:
                self.left = node
            else:
                self.left.insert(node)
        else:
            if self.right is None:
                self.right = node
            else:
                self.right.insert(node)


if __name__ == "__main__":
    root = HybridNode(1, 50, "root", 10)
    sm = pm.StateMachine()
    sm.capture(root)

    nodes = [
        HybridNode(2, 30, "L", 7),
        HybridNode(3, 70, "R", 8),
        HybridNode(4, 20, "LL", 5),
        HybridNode(5, 40, "LR", 6),
        HybridNode(6, 60, "RL", 4),
        HybridNode(7, 80, "RR", 9),
    ]

    with pm.record("Hybrid insert", sm):
        for n in nodes:
            root.insert(n)

    sm.render(path="media/container_and_links_layout.gif")
