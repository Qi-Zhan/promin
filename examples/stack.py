"""
Example: Stack (list-backed) with promin visualization.

Usage:
    uv run python examples/stack.py
"""

import promin as pm


def _stack_column_layout(targets, origin, ctx):
    out = []
    for i, child in enumerate(targets):
        out.append(
            child.with_pos(pm.Position(x=origin.pos.x, y=origin.pos.y - (i + 1) * ctx.gap_y))
        )
    return out

@(
    pm.type()
    .shape("box")
    .show(lambda s: ["Stack"])
    .links(
        pm.links()
        .items(lambda s: list(s.items))
        .layout(_stack_column_layout)
    )
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
