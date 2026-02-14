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
    NodeView,
    EdgeSpec,
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
    """ListView.render should recursively render items via View.render_value."""
    result = View.render_value([1, "hello", 3.14])
    assert result["type"] == "list"
    assert result["items"][0] == {"type": "int", "value": 1, "style": None}
    assert result["items"][1] == {"type": "str", "value": "hello", "style": None}
    assert result["items"][2] == {"type": "float", "value": 3.14, "style": None}


def test_render_value_recursive_dict():
    """DictView.render should recursively render values via View.render_value."""
    result = View.render_value({"a": 1, "b": [2, 3]})
    assert result["type"] == "dict"
    assert result["items"]["a"] == {"type": "int", "value": 1, "style": None}
    assert result["items"]["b"]["type"] == "list"


def test_render_value_unknown_type():
    """View.render_value should fall back gracefully for unregistered types."""

    class Opaque:
        pass

    result = View.render_value(Opaque())
    assert result["type"] == "unknown"


# ---------------------------------------------------------------------------
# format_label / format_value tests
# ---------------------------------------------------------------------------


def test_format_value_primitives():
    """View.format_value delegates to each View's format_label."""
    assert View.format_value(42) == "42"
    assert View.format_value(3.14) == "3.14"
    assert View.format_value("hi") == "hi"
    assert View.format_value(True) == "True"
    assert View.format_value(None) == "∅"


def test_format_value_containers():
    """format_value recursively formats lists, tuples, dicts via View dispatch."""
    assert View.format_value([1, 2, 3]) == "[1, 2, 3]"
    assert View.format_value((4, 5)) == "(4, 5)"
    assert View.format_value({"a": 1}) == "{a: 1}"


def test_format_value_nested():
    """Nested containers are recursively formatted through dispatch."""
    assert View.format_value([1, [2, 3]]) == "[1, [2, 3]]"
    assert View.format_value({"x": [1, 2]}) == "{x: [1, 2]}"


def test_format_value_unknown_type():
    """Unregistered types fall back to repr."""

    class Mystery:
        pass

    result = View.format_value(Mystery())
    assert "Mystery" in result


# ---------------------------------------------------------------------------
# RegisteredClassView tests
# ---------------------------------------------------------------------------


def test_registered_class_view_carries_node_view():
    """RegisteredClassView stores and exposes the full NodeView spec."""
    nv = NodeView(
        shape="diamond",
        label="key",
        edges=[EdgeSpec(field="left"), EdgeSpec(field="right")],
    )
    view = RegisteredClassView(nv)
    assert view.node_view is nv
    rendered = view.render(object())
    assert rendered["type"] == "registered_node"
    assert rendered["node_view"]["shape"] == "diamond"
    assert rendered["node_view"]["label"] == "key"


def test_registered_class_view_format_label():
    """RegisteredClassView.format_label reads the actual label field."""
    nv = NodeView(shape="circle", label="key")
    view = RegisteredClassView(nv)

    class Dummy:
        key = 99

    assert view.format_label(Dummy()) == "99"


def test_register_class_creates_registered_class_view():
    """@register_class should register a RegisteredClassView, not TreeView."""
    import promin as pm

    @pm.register_class(shape="box", label="val", edges=["child"])
    class TestNode:
        def __init__(self):
            self.val = 7
            self.child = None

    view = View.for_value(TestNode())
    assert isinstance(view, RegisteredClassView)
    assert view.node_view.shape == "box"
    assert view.node_view.label == "val"
    assert view.format_label(TestNode()) == "7"
