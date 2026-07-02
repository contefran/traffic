"""Tests for per-intersection signal timing (SignalPlan: offset / split / cycle)."""

import pytest

from traffic_sim import (
    build_grid_network,
    Orientation,
    TurnType,
    SignalPlan,
    FixedTimeController,
    ProtectedPhaseController,
    SignalSystem,
)


def test_plan_cycle_and_equal_split():
    plan = SignalPlan(green_times=(5.0, 5.0, 5.0, 5.0))
    assert plan.cycle == 20.0
    assert [plan.active_phase(t) for t in (0.0, 4.9, 5.0, 12.0, 19.9, 20.0)] \
        == [0, 0, 1, 2, 3, 0]


def test_plan_offset_shifts_schedule():
    base = SignalPlan(green_times=(5.0, 5.0))
    shifted = SignalPlan(green_times=(5.0, 5.0), offset=5.0)
    # The offset delays the schedule by one phase.
    for t in (0.0, 2.5, 5.0, 7.5, 10.0):
        assert shifted.active_phase(t) == base.active_phase(t - 5.0)


def test_plan_split_gives_phases_unequal_green():
    # Long H green, short V green.
    plan = SignalPlan(green_times=(15.0, 5.0))
    assert plan.cycle == 20.0
    assert plan.active_phase(14.9) == 0   # still in the long phase
    assert plan.active_phase(15.1) == 1   # into the short phase
    assert plan.active_phase(20.1) == 0   # wrapped around


def test_plan_rejects_bad_durations():
    with pytest.raises(ValueError):
        SignalPlan(green_times=())
    with pytest.raises(ValueError):
        SignalPlan(green_times=(5.0, 0.0))


def test_default_plan_reproduces_global_clock():
    # Without per-node plans, every node stays in phase (old behaviour).
    ctrl = ProtectedPhaseController(green_time=5.0)
    for t in (0.0, 5.0, 10.0, 15.0):
        p = ctrl.phase(0, t)
        assert all(ctrl.phase(n, t) == p for n in (1, 2, 3, 99))


def test_two_nodes_can_be_out_of_phase():
    ctrl = ProtectedPhaseController(green_time=5.0)
    ctrl.set_plan(7, SignalPlan(green_times=(5.0,) * 4, offset=10.0))
    # At t=0 node 0 is in phase 0 (H through); node 7, offset 10, is in phase 2.
    assert ctrl.phase(0, 0.0) == 0
    assert ctrl.phase(7, 0.0) == 2
    assert ctrl.allows(0, Orientation.HORIZONTAL, TurnType.STRAIGHT, 0.0) is True
    assert ctrl.allows(7, Orientation.HORIZONTAL, TurnType.STRAIGHT, 0.0) is False
    assert ctrl.allows(7, Orientation.VERTICAL, TurnType.STRAIGHT, 0.0) is True


def test_signalsystem_respects_per_node_offset():
    # A 4x4 grid has several interior 4-way (signalized) nodes. Offset one and
    # leave another on the default: at t=0 they permit different movements.
    net = build_grid_network(width=4, height=4, block=50.0)
    ctrl = ProtectedPhaseController(green_time=5.0)
    offset_node = net.node_id[(1, 1)]
    default_node = net.node_id[(2, 2)]
    ctrl.set_plan(offset_node, SignalPlan(green_times=(5.0,) * 4, offset=10.0))
    sig = SignalSystem(net, ctrl)
    assert sig.is_signalized(offset_node) and sig.is_signalized(default_node)

    # Default node at t=0: phase 0 -> H-through green, V-through red.
    assert sig.allows(default_node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 0.0) is True
    assert sig.allows(default_node, Orientation.VERTICAL, TurnType.STRAIGHT, 0.0) is False
    # Offset node at t=0: phase 2 -> V-through green, H-through red.
    assert sig.allows(offset_node, Orientation.VERTICAL, TurnType.STRAIGHT, 0.0) is True
    assert sig.allows(offset_node, Orientation.HORIZONTAL, TurnType.STRAIGHT, 0.0) is False


def test_fixed_time_offset_flips_orientation():
    ctrl = FixedTimeController(green_time=10.0, start=Orientation.HORIZONTAL)
    ctrl.set_plan(5, SignalPlan(green_times=(10.0, 10.0), offset=10.0))
    # Node 0 (default): H green at t=0. Node 5 (offset 10): V green at t=0.
    assert ctrl.green_orientation(0, 0.0) is Orientation.HORIZONTAL
    assert ctrl.green_orientation(5, 0.0) is Orientation.VERTICAL
