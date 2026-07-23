"""Tests for the green-wave offset preset (signals.apply_green_wave)."""

import math

from traffic_sim import (
    build_grid_network,
    Orientation,
    ProtectedPhaseController,
    SignalPlan,
    SignalState,
    SignalSystem,
    TurnType,
    apply_green_wave,
    apply_speed_scaled_yellows,
)


def _system(block=100.0):
    net = build_grid_network(width=4, height=4, block=block)
    ctrl = ProtectedPhaseController(green_time=5.0, yellow=1.5)
    return net, ctrl, SignalSystem(net, ctrl)


def test_offsets_grow_with_x_at_distance_over_speed():
    net, ctrl, signals = _system(block=100.0)
    installed = apply_green_wave(signals)
    v = net.edges[0].speed_limit  # uniform grid: one speed everywhere
    for (i, j), nid in net.node_id.items():
        assert math.isclose(installed[nid], (i * 100.0) / v)
        assert math.isclose(ctrl.plan_for(nid).offset, (i * 100.0) / v)


def test_explicit_speed_overrides_approach_speed():
    net, ctrl, signals = _system(block=100.0)
    installed = apply_green_wave(signals, speed=10.0)
    east = net.node_id[(2, 1)]
    assert math.isclose(installed[east], 20.0)


def test_vertical_wave_uses_y():
    net, ctrl, signals = _system(block=100.0)
    installed = apply_green_wave(signals, speed=10.0,
                                 orientation=Orientation.VERTICAL)
    north = net.node_id[(1, 3)]
    assert math.isclose(installed[north], 30.0)


def test_greens_and_yellows_are_preserved():
    net, ctrl, signals = _system()
    apply_speed_scaled_yellows(signals, braking=2.0)  # install long yellows
    node = net.node_id[(2, 2)]
    before = ctrl.plan_for(node)
    assert before.yellow > 1.5
    apply_green_wave(signals)
    after = ctrl.plan_for(node)
    assert after.green_times == before.green_times
    assert after.yellow == before.yellow
    assert after.offset > 0.0


def test_downstream_green_onset_lags_by_travel_time():
    # A platoon leaving node (1, 1) eastbound at the phase-0 onset should find
    # node (2, 1), one block downstream, turning green exactly on arrival.
    net, ctrl, signals = _system(block=100.0)
    apply_green_wave(signals)
    v = net.edges[0].speed_limit
    here, there = net.node_id[(1, 1)], net.node_id[(2, 1)]
    t0 = ctrl.plan_for(here).offset          # phase-0 (H-through) onset here
    travel = 100.0 / v
    # Just before the platoon arrives the downstream signal is not yet green
    # for H-through; at arrival it is.
    assert signals.state(there, Orientation.HORIZONTAL, TurnType.STRAIGHT,
                         t0 + travel) is SignalState.GREEN
    assert signals.state(there, Orientation.HORIZONTAL, TurnType.STRAIGHT,
                         t0 + travel - 0.5) is not SignalState.GREEN
