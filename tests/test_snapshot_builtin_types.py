from promin.tracing.trace import snapshot_objects
import promin as pm


def test_int_and_bool_snapshot_as_nodes():
    snaps = snapshot_objects([1, True])
    int_node, bool_node = snaps

    assert int_node["_type"] == "int"
    assert int_node["_view"]["shape"] == "box"
    assert int_node["value"] == 1

    assert bool_node["_type"] == "bool"
    assert bool_node["_view"]["shape"] == "diamond"
    assert bool_node["value"] is True


def test_list_snapshot_has_element_nodes_in_order():
    snaps = snapshot_objects([[10, 20, 30]])
    list_node = snaps[0]

    assert list_node["_type"] == "list"
    assert callable(list_node["_view"]["layout"])
    assert [n["value"] for n in list_node["elements"]] == [10, 20, 30]


def test_color_field_is_tracked_without_extra_field_list():
    @pm.register_type(
        layout=pm.RowLayout(),
        shape="circle",
        label="key",
        color_field="color",
        color_map={"red": "#CC0000", "black": "#1A1A1A"},
    )
    class _ColorNode:
        def __init__(self, key: int, color: str):
            self.key = key
            self.color = color

    node = _ColorNode(7, "red")
    snap = snapshot_objects([node])[0]
    assert snap["key"]["_type"] == "int"
    assert snap["key"]["value"] == 7
    assert snap["color"] == "red"
    assert snap["_view"]["color_field"] == "color"
