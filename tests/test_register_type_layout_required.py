import pytest

import promin as pm


def test_register_type_requires_layout():
    with pytest.raises(TypeError):
        pm.register_type(shape="circle", label="k")


def test_register_type_rejects_string_layout():
    with pytest.raises(TypeError, match="not a bare string"):
        pm.register_type(layout="tree", shape="circle", label="k")(type("N", (), {}))


def test_register_type_rejects_non_serializable_layout_params():
    with pytest.raises(TypeError, match="JSON-serializable"):
        pm.register_type(
            layout={"name": "row", "params": {"bad": {1, 2}}},
            shape="circle",
            label="k",
        )(type("N", (), {}))


def test_register_type_rejects_invalid_content_field():
    with pytest.raises(TypeError, match="content_field must reference a tracked field"):
        pm.register_type(
            layout={"name": "tree", "params": {}},
            shape="circle",
            label="k",
            content_field="missing",
        )(type("N", (), {}))
