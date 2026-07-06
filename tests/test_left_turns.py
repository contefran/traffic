"""Tests for permissive-left gap acceptance (PermissiveLeftModel)."""

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    SignalSystem,
    FixedTimeController,
    ProtectedPhaseController,
    PermissiveLeftModel,
    Orientation,
)


def _edge(net, a, b):
    for eid in net.nodes[a].out_edges:
        if net.edges[eid].v == b:
            return eid
    raise AssertionError(f"no edge {a}->{b}")


def _nodes(net):
    W = net.node_id[(0, 1)]
    X = net.node_id[(1, 1)]
    N = net.node_id[(1, 2)]
    E = net.node_id[(2, 1)]
    return W, X, N, E


def test_opposing_map_is_the_collinear_approach():
    net = build_grid_network(3, 3, block=50.0)
    W, X, N, E = _nodes(net)
    model = PermissiveLeftModel(net)
    approach = _edge(net, W, X)                       # west approach to X
    opposing_edge, straight_exit = model._opposing[approach]
    assert opposing_edge == _edge(net, E, X)          # east approach (opposite side)
    assert straight_exit == _edge(net, X, W)          # its straight-through exit


def test_no_yield_for_a_through_movement():
    net = build_grid_network(3, 3, block=50.0)
    W, X, N, E = _nodes(net)
    model = PermissiveLeftModel(net)
    signals = SignalSystem(net, FixedTimeController(green_time=30.0,
                                                    start=Orientation.HORIZONTAL))
    approach = _edge(net, W, X)
    through = _edge(net, X, E)                         # straight, not a left
    onc = Car(id=0, edge_id=_edge(net, E, X), s=48.0, v=12.0)
    assert model.must_yield(approach, through, signals, 0.1, {onc.edge_id: [onc]}) is False


def test_left_yields_when_oncoming_is_imminent_but_not_when_far():
    net = build_grid_network(3, 3, block=50.0)
    W, X, N, E = _nodes(net)
    model = PermissiveLeftModel(net)
    signals = SignalSystem(net, FixedTimeController(green_time=30.0,
                                                    start=Orientation.HORIZONTAL))
    approach = _edge(net, W, X)
    left = _edge(net, X, N)                            # turn north = left
    onc_edge = _edge(net, E, X)

    close = Car(id=0, edge_id=onc_edge, s=48.0, v=12.0)   # gap 2 m -> imminent
    assert model.must_yield(approach, left, signals, 0.1, {onc_edge: [close]}) is True

    far = Car(id=0, edge_id=onc_edge, s=5.0, v=12.0)      # gap 45 m -> accept
    assert model.must_yield(approach, left, signals, 0.1, {onc_edge: [far]}) is False

    assert model.must_yield(approach, left, signals, 0.1, {}) is False  # no oncoming


def test_inert_under_protected_phasing():
    # When a protected left has its green (phase 1), the opposing through is red,
    # so there is nothing to yield to.
    net = build_grid_network(3, 3, block=50.0)
    W, X, N, E = _nodes(net)
    model = PermissiveLeftModel(net)
    signals = SignalSystem(net, ProtectedPhaseController(green_time=5.0))  # cycle 20
    approach = _edge(net, W, X)
    left = _edge(net, X, N)
    onc_edge = _edge(net, E, X)
    close = Car(id=0, edge_id=onc_edge, s=48.0, v=12.0)
    # t=6 s is inside phase 1 (H-left green, H-through red).
    assert model.must_yield(approach, left, signals, 6.0, {onc_edge: [close]}) is False


def _left_turn_sim(with_model):
    net = build_grid_network(3, 3, block=50.0)
    W, X, N, E = _nodes(net)
    approach = _edge(net, W, X)
    a = Car(id=0, edge_id=approach, s=45.0, v=10.0)
    a.next_edge = _edge(net, X, N)                     # committed to turn left
    b = Car(id=1, edge_id=_edge(net, E, X), s=48.0, v=12.0)   # imminent oncoming
    signals = SignalSystem(net, FixedTimeController(green_time=30.0,
                                                    start=Orientation.HORIZONTAL))
    lt = PermissiveLeftModel(net) if with_model else None
    sim = TrafficSim(net, [a, b], RandomRouter(net, seed=0), signals=signals, left_turn=lt)
    return sim, a


def test_permissive_left_brakes_for_oncoming_when_model_present():
    sim_yes, a_yes = _left_turn_sim(with_model=True)
    sim_no, a_no = _left_turn_sim(with_model=False)
    sim_yes.step(0.1)
    sim_no.step(0.1)
    # With the model the left-turner yields (brakes); without it, it proceeds.
    assert a_yes.v < a_no.v


def test_permissive_left_is_inert_at_unsignalized_nodes():
    # At an *unsignalized* node right-of-way is the PriorityModel's job; the
    # permissive-left model must not fire (movement_state is GREEN there by
    # default, and applying it deadlocked elevated ramp merges — see the guard
    # in TrafficSim.step). A left-turner should proceed even with imminent
    # oncoming and the model attached.
    net = build_grid_network(3, 3, block=50.0)
    W, X, N, E = _nodes(net)
    approach = _edge(net, W, X)
    a = Car(id=0, edge_id=approach, s=45.0, v=10.0)
    a.next_edge = _edge(net, X, N)                     # committed to turn left
    b = Car(id=1, edge_id=_edge(net, E, X), s=48.0, v=12.0)   # imminent oncoming
    signals = SignalSystem(net, FixedTimeController(green_time=30.0,
                                                    start=Orientation.HORIZONTAL),
                           unsignalized_nodes={X})
    sim = TrafficSim(net, [a, b], RandomRouter(net, seed=0), signals=signals,
                     left_turn=PermissiveLeftModel(net))
    v0 = a.v
    sim.step(0.1)
    assert a.v > v0, "left-turner must not yield at an unsignalized node"
