import pytest

import promin as pm


def test_register_type_requires_layout():
    with pytest.raises(TypeError):
        pm.register_type(shape="circle", label="k")


def test_register_type_rejects_string_layout():
    with pytest.raises(TypeError, match="must be callable"):
        pm.register_type(layout="tree", shape="circle", label="k")(type("N", (), {}))


def test_register_type_rejects_dict_layout():
    with pytest.raises(TypeError, match="must be callable"):
        pm.register_type(
            layout={"name": "row", "params": {"bad": {1, 2}}},
            shape="circle",
            label="k",
        )(type("N", (), {}))


def test_register_type_rejects_invalid_content_field():
    pm.register_type(
        layout=pm.TreeLayout,
        shape="circle",
        label="k",
        content_field="missing",
    )(type("N", (), {}))


def test_register_type_rejects_data_parameter():
    with pytest.raises(TypeError):
        pm.register_type(
            layout=pm.TreeLayout,
            shape="circle",
            label="k",
            data=["x"],
        )(type("N2", (), {}))
