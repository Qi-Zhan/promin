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

### 2. Record & render

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
| `examples/rbtree.py`   | Red-black tree data structure        |

Run any example:

```bash
python examples/bst.py       # → bst_search_4.mp4, bst_insert_9.mp4
python examples/bptree.py    # → bptree_insert.mp4, bptree_search.mp4
```

## Project Structure

```
src/promin/
├── __init__.py   Public API
├── view.py       Declarative visual specs (NodeView, EdgeSpec, View dispatch)
├── trace.py      register_class, snapshot, StateMachine, record()
└── render.py     Manim-based rendering, tree layout, diff-based animation
```

## License

MIT

### Configuration

```python
scene.config(
    step_duration=0.5,   # base animation duration (seconds)
    event_pause=0.1,     # pause between events
    batch_steps=False,   # True: group iteration events into one AnimationGroup
    max_steps=50,        # cap animated iterations (rest still execute)
)
```

## Semantic Event System

All events flow through a context-local `Tracer` stored in a `ContextVar`.
Each event is a `(type, name, data, seq, timestamp)` record:

| EventType | Emitted By | Data Fields |
|---|---|---|
| `VAR_CREATED` | `Var.__init__` | `value` |
| `VAR_UPDATED` | `Var.set()`, `Var.append()` | `old_value`, `new_value` |
| `STEP_START` | `step()` | `index`, `value` |
| `STEP_END` | `step()` | `index` |
| `PHASE_START` | `phase().__enter__` | user kwargs |
| `PHASE_END` | `phase().__exit__` | user kwargs |
| `CALL` | `@trace` wrapper | `args_count`, `kwargs_keys` |
| `RETURN` | `@trace` wrapper | `has_result` |
| `EMIT` | `emit()` | user kwargs |

The `ComputationGraph` transforms the flat event stream into:
- **Variable chains** — sequential updates to the same variable are linked.
- **Step groups** — events within the same loop iteration are grouped.
- **Phase nesting** — events within phase boundaries form subtrees.

## Visual Primitives

Bindings connect variable names to visual representations. The compiler
replays events and routes each to matching bindings:

```python
scene.bind("x",        pm.NumberLineDot(range=(-1, 6)))       # scalar → dot on number line
scene.bind("loss",     pm.LivePlot(color="red"))              # emit → growing scatter plot
scene.bind("gradient", pm.GradientArrow(number_line=x_dot))  # emit → direction arrow
scene.bind("tokens",   pm.DataTable(max_rows=15))             # list → visual table
scene.bind("counter",  pm.ValueDisplay(fmt=".4f"))            # scalar → text readout
scene.bind("iter",     pm.StepCounter())                      # iteration number
```

### Built-in Visuals

| Visual | Best For | Listens To |
|---|---|---|
| `NumberLineDot` | Scalars, positions, parameters | `VAR_CREATED`, `VAR_UPDATED` |
| `LivePlot` | Loss curves, metrics, convergence | `EMIT` (with `value`), `VAR_UPDATED` |
| `ValueDisplay` | Monitoring scalars, counters | `VAR_CREATED`, `VAR_UPDATED`, `EMIT` |
| `DataTable` | Token lists, arrays, dicts | `VAR_UPDATED` (list/dict) |
| `GradientArrow` | Gradient direction + magnitude | `EMIT` (with `value`, `at`) |
| `StepCounter` | Iteration number | `STEP_START` |

### Custom Visuals

Extend `pm.Visual` to create domain-specific visuals:

```python
class HeatmapGrid(pm.Visual):
    def __init__(self, rows, cols, **kwargs):
        self.rows = rows
        self.cols = cols
        # store config — no Manim imports in __init__

    def create_mobjects(self) -> list:
        # Called once — return initial Manim mobjects
        from manim import Square, VGroup
        ...
        return [grid]

    def animate_event(self, event: pm.Event) -> list:
        # Called per matching event — return Manim animations
        from manim import FadeToColor
        ...
        return [FadeToColor(cell, new_color)]
```

## Examples

### Gradient Descent

```bash
python examples/gradient_descent.py           # dry-run: print trace
python examples/gradient_descent.py --render  # produce video
manim -pql examples/gradient_descent.py GradientDescentScene  # via Manim CLI
```

### Tokenizer

```bash
python examples/tokenizer_viz.py              # dry-run: print trace
python examples/tokenizer_viz.py --render     # produce video
manim -pql examples/tokenizer_viz.py TokenizerScene
```

## Key Properties

- **Manim is lazy** — tracing and `preview()` work without Manim installed. Manim is only imported at render time.
- **Var is transparent** — arithmetic operators delegate to the underlying value, so `Var` works in normal expressions (`x + 1`, `x * 2`).
- **Adaptive pacing** — the compiler auto-slows early iterations (for clarity) and speeds up later ones.
- **Batch mode** — `batch_steps=True` groups all events in one iteration into a single `AnimationGroup` for faster playback on high iteration counts.

## Requirements

- Python 3.10+
- [Manim Community](https://docs.manim.community/) (only for rendering; dry-run works without it)

## License

MIT
