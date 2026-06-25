"""Movement classification, protected-phase signals, and unsignalized nodes."""

import pytest

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    Orientation,
    TurnType,
    ProtectedPhaseController,
    SignalSystem,
)
from traffic_sim.signals import turn_type


def edge_between(net, a, b):
    """Edge id from grid point a=(i,j) to b=(i,j)."""
    ua, ub = net.node_id[a], net.node_id[b]
    return next(e.id for e in net.edges if e.u == ua and e.v == ub)


def test_turn_type_classifies_all_movements():
    net = build_grid_network(width=3, height=3, block=50.0)
    # Approach the centre heading east: (0,1) -> (1,1).
    approach = edge_between(net, (0, 1), (1, 1))
    assert turn_type(net, approach, edge_between(net, (1, 1), (2, 1))) is TurnType.STRAIGHT
    assert turn_type(net, approach, edge_between(net, (1, 1), (1, 2))) is TurnType.LEFT   # north
    assert turn_type(net, approach, edge_between(net, (1, 1), (1, 0))) is TurnType.RIGHT  # south
    assert turn_type(net, approach, edge_between(net, (1, 1), (0, 1))) is TurnType.UTURN


def test_protected_controller_phase_schedule():
    ctrl = ProtectedPhaseController(green_time=5.0)
    # Phase 0: H through only.
    assert ctrl.allows(0, Orientation.HORIZONTAL, TurnType.STRAIGHT, 0.0) is True
    assert ctrl.allows(0, Orientation.HORIZONTAL, TurnType.LEFT, 0.0) is False
    assert ctrl.allows(0, Orientation.VERTICAL, TurnType.STRAIGHT, 0.0) is False
    # Phase 1: H left only.
    assert ctrl.allows(0, Orientation.HORIZONTAL, TurnType.LEFT, 5.0) is True
    assert ctrl.allows(0, Orientation.HORIZONTAL, TurnType.STRAIGHT, 5.0) is False
    # Phase 2/3: V through, then V left.
    assert ctrl.allows(0, Orientation.VERTICAL, TurnType.STRAIGHT, 10.0) is True
    assert ctrl.allows(0, Orientation.VERTICAL, TurnType.LEFT, 15.0) is True


def test_through_and_left_never_share_a_phase():
    ctrl = ProtectedPhaseController(green_time=5.0)
    for orient in (Orientation.HORIZONTAL, Orientation.VERTICAL):
        for t in [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]:
            through = ctrl.allows(0, orient, TurnType.STRAIGHT, t)
            left = ctrl.allows(0, orient, TurnType.LEFT, t)
            assert not (through and left)


def test_protected_left_waits_for_its_phase_then_turns():
    net = build_grid_network(width=3, height=3, block=50.0)
    sig = SignalSystem(net, ProtectedPhaseController(green_time=5.0))
    approach = edge_between(net, (0, 1), (1, 1))      # heading east
    left_turn = edge_between(net, (1, 1), (1, 2))     # to the north = left
    length = net.edges[approach].length

    car = Car(id=0, edge_id=approach, s=length - 12.0, v=10.0, next_edge=left_turn)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=1), signals=sig)

    for _ in range(40):   # t -> 4s, phase 0 (H through): left turn is RED
        sim.step(0.1)
    assert car.edge_id == approach   # still waiting, has not turned
    assert car.v < 1.0

    for _ in range(40):   # t -> 8s, phase 1 (H left): now allowed
        sim.step(0.1)
    assert car.edge_id == left_turn  # completed the protected left


def test_unsignalized_node_allows_every_movement():
    net = build_grid_network(width=3, height=3, block=50.0)
    centre = net.node_id[(1, 1)]
    sig = SignalSystem(net, ProtectedPhaseController(green_time=5.0),
                       unsignalized_nodes={centre})
    assert sig.is_signalized(centre) is False
    approach = edge_between(net, (0, 1), (1, 1))
    left_turn = edge_between(net, (1, 1), (1, 2))
    # Allowed at every phase, including when a protected light would be red.
    for t in [0.0, 5.0, 10.0, 15.0]:
        assert sig.allows_movement(approach, left_turn, t) is True


def test_unsignalized_node_does_not_stop_cars():
    net = build_grid_network(width=3, height=3, block=50.0)
    centre = net.node_id[(1, 1)]
    sig = SignalSystem(net, ProtectedPhaseController(green_time=5.0),
                       unsignalized_nodes={centre})
    approach = edge_between(net, (0, 1), (1, 1))
    left_turn = edge_between(net, (1, 1), (1, 2))
    car = Car(id=0, edge_id=approach, s=net.edges[approach].length - 5.0, v=10.0,
              next_edge=left_turn)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=1), signals=sig)
    for _ in range(40):
        sim.step(0.1)
    assert car.edge_id == left_turn  # flowed straight through, no waiting
