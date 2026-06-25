from traffic_sim import build_grid_network
from traffic_sim.signals import (
    Orientation,
    edge_orientation,
    FixedTimeController,
    SignalSystem,
)


def test_edge_orientation_classifies_grid_edges():
    net = build_grid_network(width=3, height=3, block=10.0)
    horiz = vert = 0
    for e in net.edges:
        o = edge_orientation(net, e.id)
        if o is Orientation.HORIZONTAL:
            assert net.nodes[e.u].j == net.nodes[e.v].j
            horiz += 1
        else:
            assert net.nodes[e.u].i == net.nodes[e.v].i
            vert += 1
    assert horiz > 0 and vert > 0


def test_fixed_time_controller_alternates():
    ctrl = FixedTimeController(green_time=10.0, start=Orientation.HORIZONTAL)
    assert ctrl.green_orientation(0, 0.0) is Orientation.HORIZONTAL
    assert ctrl.green_orientation(0, 9.9) is Orientation.HORIZONTAL
    assert ctrl.green_orientation(0, 10.0) is Orientation.VERTICAL
    assert ctrl.green_orientation(0, 20.0) is Orientation.HORIZONTAL


def test_signal_system_green_red_at_intersection():
    net = build_grid_network(width=3, height=3, block=50.0)
    sig = SignalSystem(net, FixedTimeController(green_time=10.0, start=Orientation.HORIZONTAL))

    # Find a horizontal and a vertical edge that both feed a signalized node.
    interior = next(n.id for n in net.nodes if sig.is_signalized(n.id))
    in_edges = net.nodes[interior].in_edges
    h_edge = next(e for e in in_edges if edge_orientation(net, e) is Orientation.HORIZONTAL)
    v_edge = next(e for e in in_edges if edge_orientation(net, e) is Orientation.VERTICAL)

    # t=0: horizontal green, vertical red. After green_time: swapped.
    assert sig.is_green(h_edge, 0.0) is True
    assert sig.is_green(v_edge, 0.0) is False
    assert sig.is_green(h_edge, 10.0) is False
    assert sig.is_green(v_edge, 10.0) is True


def test_grid_corner_is_a_real_intersection():
    # In a 2D grid even a corner has one horizontal and one vertical road
    # meeting, so it is signalized.
    net = build_grid_network(width=3, height=3, block=50.0)
    sig = SignalSystem(net, FixedTimeController())
    assert sig.is_signalized(net.node_id[(0, 0)]) is True


def test_single_orientation_line_is_never_signalized():
    # A 1xN line has only vertical edges: no cross traffic, always green.
    net = build_grid_network(width=1, height=4, block=50.0)
    sig = SignalSystem(net, FixedTimeController())
    for node in net.nodes:
        assert sig.is_signalized(node.id) is False
    for e in net.edges:
        assert sig.is_green(e.id, 0.0) is True
        assert sig.is_green(e.id, 10.0) is True
