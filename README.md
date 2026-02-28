# promin: Automatic program visualization powered by Manim.

## Setup (uv)

```bash
uv sync
uv sync --group dev
```

## Quick Start

```python
import promin as pm

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
class BSTNode:
    ...
```

## Core API

- `pm.type(...).shape(...).show(...).fill(...).text(...).links(...)`
- `pm.links().items(...).hints(...).layout(...)`
- Built-in layouts: `pm.tree`, `pm.row(...)`, `pm.column()`, `pm.radial(...)`
- Position primitives for custom layouts: `pm.Position`, `pm.Anchor`

## Example

```python
sm = pm.StateMachine()
sm.capture(root)

with pm.record("Insert", sm):
    root.insert(9)

sm.render(path="media/bst_insert_9.gif")
```

See:
- `examples/bst.py`
- `examples/rbtree.py`
- `examples/stack.py`
- `examples/autograd.py`
