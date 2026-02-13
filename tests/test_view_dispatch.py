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
