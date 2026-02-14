# promin

**Automatic algorithm visualization powered by Manim.**

Decorate your data-structure class, run an algorithm, get a video — no
animation code required.  Change the input and the video changes with it.

```
register_class → record() → StateMachine → .render("out.mp4")
```

## Installation

```bash
# Requires Python ≥ 3.10 and Manim ≥ 0.18
pip install -e .
```

> **Tip:** If you use conda, create an environment with Manim first:
> ```bash
> conda create -n manim python=3.12 manim -c conda-forge
> conda activate manim
> pip install -e .
> ```

## Quick Start

### 1. Declare your data structure

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import promin as pm

RED, BLACK, NIL_KEY = "red", "black", "NIL"

@pm.register_class(
    shape="circle",
    label="key",
    edges=[
        pm.EdgeSpec(field="left",  direction="left"),
        pm.EdgeSpec(field="right", direction="right"),
    ],
    data=["color"],
    color_field="color",
    color_map={"red": "#CC0000", "black": "#1A1A1A"},
    skip_if=lambda node: node.key == NIL_KEY,
)
@dataclass
class RBNode:
    key: int | str
    color: str = RED
    left:  Optional[RBNode] = None
    right: Optional[RBNode] = None
    parent: Optional[RBNode] = None
```

The full Red-Black Tree implementation (rotations, insert fix-up, delete, etc.)
lives in [`examples/rbtree.py`](examples/rbtree.py).

### 2. Record & Render

```python
t = RedBlackTree()
keys = [7, 3, 18, 10, 22, 8]
t.insert(keys[0])

sm = pm.StateMachine()
sm.capture(t.root)

with pm.record("RBTree insert", sm):
    for k in keys[1:]:
        t.insert(k)

sm.render(
    path="media/rbtree_insert.mp4",
    config=pm.RenderConfig(background_color="#F5F0EB"),
)
```

`record()` uses `sys.settrace` to automatically capture a state snapshot every
time the algorithm visits a registered-class object.  The renderer diffs
consecutive snapshots and animates only the changes.


<video src="media/rbtree_insert.mp4" controls width="100%"></video>

## How It Works

1. **`@register_class`** — Tells promin which fields to track and how to draw
   each node (shape, label field, edge fields).
2. **`StateMachine.capture(root)`** — Registers live objects as snapshot roots.
3. **`record(name, sm)`** — Traces execution via `sys.settrace`, capturing a
   `State` every time a registered object is read or mutated.
4. **`sm.render(path)`** — Generates a self-contained Manim scene and renders
   it to video.  The renderer is fully data-driven — shapes, labels, and edges
   are all derived from the `_view` metadata embedded in each snapshot.

## `register_class` Parameters

| Parameter   | Type                        | Description                                    |
|-------------|---------------------------- |------------------------------------------------|
| `shape`     | `str`                       | `"circle"`, `"box"`, or `"diamond"`            |
| `label`     | `str`                       | Field name shown as text inside the shape       |
| `edges`     | `list[str \| EdgeSpec]`     | Fields that are connections to other nodes       |
| `data`      | `list[str]`                 | Extra tracked fields (not rendered as edges)     |
| `type_name` | `str`                       | Display name (defaults to class name)            |

### `EdgeSpec`

For fine-grained control over edge rendering:

```python
from promin import EdgeSpec

@pm.register_class(
    shape="box",
    label="keys",
    edges=[
        EdgeSpec(field="left",  direction="left",  style="solid"),
        EdgeSpec(field="right", direction="right", style="dashed"),
    ],
)
class MyNode: ...
```

- **`direction`** — `"auto"`, `"left"`, `"right"`, `"down"`, `"up"`
- **`style`** — `"solid"`, `"dashed"`, `"dotted"`

### List Edges

If an edge field holds a **list** of registered objects (e.g. a B+ tree's
`children`), each list element becomes a separate child edge:

```python
@pm.register_class(
    shape="box",
    label="keys",
    edges=[EdgeSpec(field="children")],
)
class BPInternal:
    def __init__(self):
        self.keys: list[int] = []
        self.children: list = []
```

## Examples

| File                    | Description                          | Output                       |
|-------------------------|--------------------------------------|------------------------------|
| `examples/rbtree.py`   | Red-Black tree insert                | `media/rbtree_insert.mp4`    |
| `examples/bst.py`      | Binary search tree insert & search   | `media/bst_*.mp4`            |
| `examples/bptree.py`   | B+ tree insert & search              | `media/bptree_*.mp4`         |

Run any example (videos are written to `media/`):

```bash
python examples/rbtree.py    # → media/rbtree_insert.mp4
python examples/bst.py       # → media/bst_search_4.mp4, media/bst_insert_9.mp4
python examples/bptree.py    # → media/bptree_insert.mp4, media/bptree_search.mp4
```
