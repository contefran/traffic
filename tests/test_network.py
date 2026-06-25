import math

from traffic_sim import build_grid_network


def test_grid_node_count():
    net = build_grid_network(width=4, height=3, block=50.0)
    assert len(net.nodes) == 12
    assert net.node_id[(0, 0)] == 0
    assert net.node_id[(3, 2)] == 11


def test_grid_edge_count_is_two_way():
    # 4x3 grid: 3*3 horizontal + 4*2 vertical adjacencies, each two-way.
    net = build_grid_network(width=4, height=3, block=50.0)
    adjacencies = (4 - 1) * 3 + 4 * (3 - 1)
    assert len(net.edges) == adjacencies * 2


def test_edge_endpoints_have_block_length():
    net = build_grid_network(width=2, height=2, block=50.0)
    for e in net.edges:
        assert math.isclose(e.length, 50.0)


def test_point_on_edge_interpolates():
    net = build_grid_network(width=2, height=1, block=50.0)
    e = net.edges[0]
    start = net.point_on_edge(e.id, 0.0)
    mid = net.point_on_edge(e.id, e.length / 2)
    end = net.point_on_edge(e.id, e.length)
    assert start == (net.nodes[e.u].x, net.nodes[e.u].y)
    assert end == (net.nodes[e.v].x, net.nodes[e.v].y)
    assert math.isclose(mid[0], (start[0] + end[0]) / 2)


def test_in_out_edges_are_consistent():
    net = build_grid_network(width=3, height=3, block=10.0)
    for e in net.edges:
        assert e.id in net.nodes[e.u].out_edges
        assert e.id in net.nodes[e.v].in_edges
