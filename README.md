# promin: Automatic program visualization powered by Manim.

## Setup (uv)

```bash
# Install project + runtime dependencies
uv sync

# Install development dependencies (pytest, etc.)
uv sync --group dev
```

> Requires Python >= 3.10. If you don't have `uv` yet: https://docs.astral.sh/uv/

## Common Commands

```bash
# Run tests
uv run -m pytest

# Run an example (small output)
uv run python examples/bst.py
```

## Quick Start

### 1. Declare your data structure

```python
import promin as pm

@pm.register_type(
    layout={"name": "tree", "params": {}},
    shape="circle",
    label="key",
    edges=["left", "right"],
)
class BSTNode:
    def __init__(self, key: int):
        self.key = key
        self.left = None
        self.right = None
```

完整示例见 [`examples/bst.py`](examples/bst.py)。

### 2. Record & Render

```python
root = BSTNode(5)
root.left = BSTNode(3)
root.right = BSTNode(7)

sm = pm.StateMachine()
sm.capture(root)

with pm.record("Search for 4", sm):
    pass

sm.render(
    path="media/bst_search_4.gif",
)
```

![BST Search](media/bst_search_4.gif)

## How It Works

1. **`@register_type`** — Tells promin which fields to track and how to draw
   each node (shape, label field, edge fields).
2. **`StateMachine.capture(root)`** — Registers live objects as snapshot roots.
3. **`record(name, sm)`** — Traces execution via `sys.settrace`, capturing a
   `State` every time a registered object is read or mutated.
4. **`sm.render(path)`** — Generates a self-contained Manim scene and renders
   it to video.  The renderer is fully data-driven — shapes, labels, and edges
   are all derived from the `_view` metadata embedded in each snapshot.

## `register_type` Parameters

`layout` is required. Promin has **no default layout fallback**.

| Parameter   | Type                        | Description                                    |
|-------------|---------------------------- |------------------------------------------------|
| `layout`    | `dict`                      | Required. `{"name": "<layout>", "params": {...}}` |
| `shape`     | `str \| None`               | `"circle"`, `"box"`, `"diamond"`, or `None` (transparent wrapper) |
| `label`     | `str`                       | Field name shown as text inside the shape       |
| `edges`     | `list[str \| EdgeSpec]`     | Fields that are connections to other nodes       |
| `data`      | `list[str]`                 | Extra tracked fields (not rendered as edges)     |
| `type_name` | `str`                       | Display name (defaults to class name)            |

### Override Built-in Type Views (`list`)

You can override built-in types with the same API:

```python
import promin as pm

pm.register_type(
    list,
    layout={"name": "row", "params": {"wrap": True, "columns": 8}},
    shape="box",
    label="size",
    data=["size"],
    label_resolver=lambda v: len(v),
    data_resolver=lambda v: {"size": len(v)},
    children_resolver=lambda v: {"elements": list(v)},
)
```

If `layout` is missing, `register_type` raises `TypeError` immediately.

### Custom Layout Plugin

```python
import promin as pm

def stack_column(ctx: pm.LayoutContext) -> pm.LayoutResult:
    positions = {}
    for i, child in enumerate(ctx.children):
        cid = child.get("node_id")
        if cid is not None:
            positions[cid] = (0.0, -(i + 1) * ctx.gap_y)
    return pm.LayoutResult(positions=positions)

pm.register_layout("stack_column", stack_column)
```

If you also want to override formatting/render dispatch, use
`register_value_view`. When the custom view provides `type_view_spec()`,
it will also replace the snapshot-side `TypeViewSpec` for that type.

### `EdgeSpec`

For fine-grained control over edge rendering:

```python
from promin import EdgeSpec

@pm.register_type(
    layout={"name": "tree", "params": {}},
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
- **`layout`** — optional per-edge layout override: `{"name": "...", "params": {...}}`

### List Edges

If an edge field holds a **list** of registered objects (e.g. a B+ tree's
`children`), each list element becomes a separate child edge:

```python
@pm.register_type(
    layout={"name": "tree", "params": {}},
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
| `examples/rbtree.py`   | Red-Black tree insert                | `media/rbtree_insert.gif`    |
| `examples/bst.py`      | Binary search tree insert & search   | `media/bst_*.gif`            |
| `examples/bptree.py`   | B+ tree insert & search              | `media/bptree_*.gif`         |

Run any example (videos are written to `media/`):

```bash
uv run python examples/rbtree.py    # → media/rbtree_insert.gif
uv run python examples/bst.py       # → media/bst_search_4.gif, media/bst_insert_9.gif
uv run python examples/bptree.py    # → media/bptree_insert.gif, media/bptree_search.gif
```
