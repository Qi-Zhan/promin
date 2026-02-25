"""
Example: Stack (list-backed) with promin visualization.

Usage:
    uv run python examples/stack.py
"""

import promin as pm


def _stack_column_layout(ctx: pm.LayoutContext) -> pm.LayoutResult:
    positions: dict[int, tuple[float, float]] = {}
    for i, child in enumerate(ctx.children):
        cid = child.get("node_id")
        if cid is None:
            continue
        positions[cid] = (0.0, -(i + 1) * ctx.gap_y)
    return pm.LayoutResult(positions=positions)

@pm.register_type(
    layout=_stack_column_layout,
    shape="box",
    label="name",
    edges=[
        pm.EdgeSpec(
            field="elements",
            direction="down",
            style="solid",
            layout=_stack_column_layout,
        )
    ],
    label_resolver=lambda s: "Stack",
    children_resolver=lambda s: {"elements": list(s.items)},
)
class Stack:
    def __init__(self):
        self.items: list[int] = []

    def push(self, value: int) -> None:
        self.items.append(value)

    def pop(self) -> int:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self) -> int:
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def is_empty(self) -> bool:
        return len(self.items) == 0


if __name__ == "__main__":
    stack = Stack()
    sm = pm.StateMachine()
    sm.capture(stack)

    with pm.record("Stack operations", sm):
        stack.push(10)
        stack.push(20)
        stack.push(30)
        stack.peek()
        stack.pop()
        stack.push(40)
        stack.pop()
        stack.pop()
        stack.push(50)
        stack.peek()

    sm.render(
        path="media/stack_ops.gif"
    )
