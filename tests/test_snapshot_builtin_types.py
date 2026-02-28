from promin.tracing.trace import snapshot_objects
import promin as pm


def test_int_and_bool_snapshot_as_nodes():
    snaps = snapshot_objects([1, True])
    int_node, bool_node = snaps

    assert int_node["_type"] == "int"
    assert int_node["_view"]["container"]["shape"] == "box"
    assert int_node["value"] == 1

    assert bool_node["_type"] == "bool"
    assert bool_node["_view"]["container"]["shape"] == "diamond"
    assert bool_node["value"] is True


def test_list_snapshot_has_element_nodes_in_order():
    snaps = snapshot_objects([[10, 20, 30]])
    list_node = snaps[0]

    assert list_node["_type"] == "list"
    assert callable(list_node["_view"]["links"]["layout"])
    assert [n["target"]["value"] for n in list_node["__links"]] == [10, 20, 30]


def test_color_field_is_tracked():
    @(
        pm.type()
        .shape("circle")
        .show(lambda n: [n.key])
        .fill(lambda n: n.color, map={"red": "#CC0000", "black": "#1A1A1A"})
    )
    class _ColorNode:
        def __init__(self, key: int, color: str):
            self.key = key
            self.color = color

    node = _ColorNode(7, "red")
    snap = snapshot_objects([node])[0]
    assert snap["__content"] == [7]
    assert snap["__color"] == "red"
    assert snap["_view"]["container"]["color_field"] == "__color"
