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
```

### 2. Record & Render

```python
root = BSTNode(5)
for k in [3, 7, 1, 4, 6, 8]:
    root.insert(k)

sm = pm.StateMachine()
sm.capture(root)

with pm.record("Search for 4", sm):
    root.search(4)

sm.render(path="bst_search.mp4")
```

`record()` uses `sys.settrace` to automatically capture a state snapshot every
time the algorithm visits a registered-class object.  The renderer diffs
consecutive snapshots and animates only the changes.

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

| File                    | Description                          |
|-------------------------|--------------------------------------|
| `examples/bst.py`      | Binary search tree insert & search   |
| `examples/bptree.py`   | B+ tree insert & search              |

Run any example:

```bash
python examples/bst.py       # → bst_search_4.mp4, bst_insert_9.mp4
python examples/bptree.py    # → bptree_insert.mp4, bptree_search.mp4
```
