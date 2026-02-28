import pytest

import promin as pm


def test_type_builder_rejects_invalid_container_shape():
    with pytest.raises(TypeError):
        pm.type().shape(123)(type("N", (), {}))


def test_type_builder_rejects_string_links_layout():
    with pytest.raises(TypeError, match=r"links\.layout must be callable"):
        pm.type().show(lambda n: [n.k]).links(pm.links().items(lambda n: [n.k]).layout("tree"))(
            type("N", (), {})
        )


def test_links_builder_rejects_non_callable_items():
    with pytest.raises(TypeError, match=r"links\.items resolver must be callable"):
        pm.links().items("x")


def test_links_builder_rejects_invalid_hints_type():
    with pytest.raises(TypeError, match=r"links\.hints must be list\[str\] or callable"):
        pm.links().hints("x")


def test_type_builder_rejects_non_builder_links_config():
    with pytest.raises(TypeError, match=r"expects pm\.links\(\) builder"):
        pm.type().show(lambda n: [n.k]).links(lambda n: [n.k])(type("NBad", (), {}))


def test_type_builder_rejects_non_callable_show():
    with pytest.raises(TypeError, match=r"container\.content must be callable"):
        pm.type().shape("circle").show("k")(type("N2", (), {}))


def test_type_builder_rejects_non_callable_text_color():
    with pytest.raises(TypeError, match=r"container\.text_color must be callable"):
        (
            pm.type()
            .shape("circle")
            .show(lambda n: [n.k])
            .text("c")
        )(type("N3", (), {}))
