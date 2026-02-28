import pytest

from promin.view import (
    View,
    IntView,
    FloatView,
    ListView,
    DictView,
    TupleView,
    StrView,
    BoolView,
    NoneView,
    SetView,
    RegisteredClassView,
    TypeViewSpec,
    container,
    links,
)


def test_view_dispatch_builtin_types():
    assert isinstance(View.for_value(1), IntView)
    assert isinstance(View.for_value(1.0), FloatView)
    assert isinstance(View.for_value([]), ListView)
    assert isinstance(View.for_value({}), DictView)
    assert isinstance(View.for_value(()), TupleView)
    assert isinstance(View.for_value("hello"), StrView)
    assert isinstance(View.for_value(True), BoolView)
    assert isinstance(View.for_value(None), NoneView)
    assert isinstance(View.for_value(set()), SetView)


def test_view_dispatch_unknown_type():
    class Custom:
        pass

    with pytest.raises(TypeError):
        View.for_value(Custom())


def test_render_value_recursive_list():
    result = View.render_value([1, "hello", 3.14])
    assert result["type"] == "list"
    assert result["items"][0] == {"type": "int", "value": 1, "style": None}
    assert result["items"][1] == {"type": "str", "value": "hello", "style": None}
    assert result["items"][2] == {"type": "float", "value": 3.14, "style": None}


def test_format_value_primitives():
    assert View.format_value(42) == "42"
    assert View.format_value(3.14) == "3.14"
    assert View.format_value("hi") == "hi"
    assert View.format_value(True) == "True"
    assert View.format_value(None) == "∅"


def test_registered_class_view_carries_type_view_spec():
    spec = TypeViewSpec(
        container=container(shape="diamond", content=lambda o: [o.key]),
        links=links().items(lambda o: [o.left, o.right]).hints(["left", "right"]).build(),
    )
    view = RegisteredClassView(spec)
    assert view.type_view is spec
    rendered = view.render(object())
    assert rendered["type"] == "registered_node"
    assert rendered["node_view"]["container"]["shape"] == "diamond"


def test_register_type_creates_registered_class_view():
    import promin as pm

    @(
        pm.type()
        .shape("box")
        .show(lambda n: [n.val])
        .links(pm.links().items(lambda n: [n.child]).layout(pm.tree))
    )
    class TestNode:
        def __init__(self):
            self.val = 7
            self.child = None

    view = View.for_value(TestNode())
    assert isinstance(view, RegisteredClassView)
    assert view.type_view.container.shape == "box"
    assert view.format_label(TestNode()) == "7"


def test_register_type_supports_lambda_content_and_links():
    import promin as pm
    from promin.tracing.trace import snapshot_objects

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
    class _N:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    root = _N(1)
    root.left = _N(2)
    snap = snapshot_objects([root])[0]
    assert snap["_view"]["container"]["content_field"] == "__content"
    assert "__content" in snap
    assert "__links" in snap
