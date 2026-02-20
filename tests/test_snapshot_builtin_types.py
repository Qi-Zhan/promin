from promin.trace import snapshot_objects


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
    assert list_node["_view"]["layout"]["name"] == "row"
    assert [n["value"] for n in list_node["elements"]] == [10, 20, 30]
