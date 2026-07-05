"""Tests for the yellow/clearance phase on traffic signals."""

import math

from traffic_sim import (
    build_grid_network,
    build_city_grid,
    Car,
    ShortestPathRouter,
    TrafficSim,
    ProtectedPhaseController,
    SignalSystem,
    SignalPlan,
    SignalState,
    Orientation,
    TurnType,
    PriorityModel,
    MetricsCollector,
    kmh_to_ms,
)


def test_signal_plan_yellow_extends_cycle_and_reports_states():
    plan = SignalPlan((5.0, 5.0), yellow=2.0)   # 5 green + 2 yellow, twice
    assert math.isclose(plan.cycle, 14.0)
    assert plan.phase_state(0.0) == (0, SignalState.GREEN)
    assert plan.phase_state(4.9) == (0, SignalState.GREEN)
    assert plan.phase_state(5.5) == (0, SignalState.YELLOW)   # phase 0 yellow: 5..7
    assert plan.phase_state(7.5) == (1, SignalState.GREEN)    # phase 1 green: 7..12
    assert plan.phase_state(12.5) == (1, SignalState.YELLOW)  # phase 1 yellow: 12..14


def test_no_yellow_is_backward_compatible():
    plan = SignalPlan((5.0, 5.0))               # yellow defaults to 0
    assert math.isclose(plan.cycle, 10.0)
    assert plan.phase_state(0.0) == (0, SignalState.GREEN)
    assert plan.phase_state(5.5) == (1, SignalState.GREEN)    # no yellow band at all
    assert plan.active_phase(5.5) == 1


def test_controller_state_is_green_then_yellow_then_red():
    ctrl = ProtectedPhaseController(green_time=5.0, yellow=2.0)  # cycle 4*(5+2)=28
    # H-through is served by phase 0 (green 0..5, yellow 5..7); red afterwards.
    node = 0
    assert ctrl.state(node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 1.0) is SignalState.GREEN
    assert ctrl.state(node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 6.0) is SignalState.YELLOW
    assert ctrl.state(node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 10.0) is SignalState.RED
    # allows() means a full green only.
    assert ctrl.allows(node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 1.0) is True
    assert ctrl.allows(node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 6.0) is False


def test_signalsystem_movement_state_green_at_unsignalized_node():
    net = build_grid_network(3, 3, block=50.0)
    interior = net.node_id[(1, 1)]
    sig = SignalSystem(net, ProtectedPhaseController(green_time=5.0, yellow=2.0),
                       unsignalized_nodes={interior})
    # A movement into the forced-unsignalized node is always green.
    in_edge = _edge_between(net, net.node_id[(0, 1)], interior)
    out_edge = _edge_between(net, interior, net.node_id[(2, 1)])
    assert sig.movement_state(in_edge, out_edge, 6.0) is SignalState.GREEN


def test_signalsystem_movement_state_follows_controller():
    net = build_grid_network(3, 3, block=50.0)
    interior = net.node_id[(1, 1)]
    sig = SignalSystem(net, ProtectedPhaseController(green_time=5.0, yellow=2.0))
    in_edge = _edge_between(net, net.node_id[(0, 1)], interior)     # horizontal approach
    out_edge = _edge_between(net, interior, net.node_id[(2, 1)])    # H-through
    assert sig.movement_state(in_edge, out_edge, 1.0) is SignalState.GREEN
    assert sig.movement_state(in_edge, out_edge, 6.0) is SignalState.YELLOW
    assert sig.movement_state(in_edge, out_edge, 10.0) is SignalState.RED


def _crashes(green, yellow):
    net = build_city_grid(8, 8, 150.0, seed=1, jitter=0.22, one_way_prob=0.15,
                          drop_prob=0.12, arterial_every=3, arterial_speed=kmh_to_ms(70))
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(60)]
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=42),
                     signals=SignalSystem(net, ProtectedPhaseController(green_time=green, yellow=yellow)),
                     priority=PriorityModel(net), metrics=MetricsCollector())
    for _ in range(800):
        sim.step(0.1)
    return sim.crashes


def test_yellow_reduces_crashes_at_equal_cycle():
    # Same 20 s cycle: a clearance interval lets dilemma-zone cars clear instead
    # of crashing the stop line, so crashes drop sharply.
    no_yellow = _crashes(green=5.0, yellow=0.0)
    with_yellow = _crashes(green=4.0, yellow=1.0)
    assert with_yellow < no_yellow


def _edge_between(net, a, b):
    for eid in net.nodes[a].out_edges:
        if net.edges[eid].v == b:
            return eid
    raise AssertionError(f"no edge {a}->{b}")
