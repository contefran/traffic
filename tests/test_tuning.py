"""Tests for the flat-vector parameter adapter (tuning.ParameterSpace)."""

import math

import pytest

from traffic_sim import (
    build_grid_network,
    ParameterSpace,
    ProtectedPhaseController,
    ShortestPathRouter,
    SignalPlan,
    SignalSystem,
    apply_speed_scaled_yellows,
)


def _space(**kwargs):
    net = build_grid_network(width=4, height=4, block=50.0)
    ctrl = ProtectedPhaseController(green_time=5.0, yellow=1.5)
    signals = SignalSystem(net, ctrl)
    return net, ctrl, signals, ParameterSpace(signals, **kwargs)


def test_layout_is_consistent_and_deterministic():
    net, ctrl, signals, space = _space()
    # On a full two-way 4x4 grid every node mixes both orientations, so all 16
    # are signalized; 4 phases + offset each.
    assert len(space.nodes) == 16
    assert space.dim == 16 * 5
    assert len(space.labels()) == space.dim == len(space.bounds())
    # Rebuilding the space gives the same layout.
    assert ParameterSpace(signals).labels() == space.labels()


def test_vector_roundtrip_is_identity():
    net, ctrl, signals, space = _space()
    ctrl.set_plan(space.nodes[0], SignalPlan((4.0, 6.0, 5.0, 5.0),
                                             offset=7.0, yellow=1.5))
    before = space.vector()
    space.apply(before)
    assert space.vector() == before
    # And the tweaked plan survived the round trip.
    plan = ctrl.plan_for(space.nodes[0])
    assert plan.green_times == (4.0, 6.0, 5.0, 5.0) and plan.offset == 7.0


def test_apply_offset_desynchronizes_node():
    net, ctrl, signals, space = _space()
    vec = space.vector()
    # Give the first node a half-cycle offset: its phase at t=0 must now differ
    # from an untouched node's.
    vec[space.labels().index(f"n{space.nodes[0]}.offset")] = \
        ctrl.default_plan.cycle / 2.0
    space.apply(vec)
    assert ctrl.phase(space.nodes[0], 0.0) != ctrl.phase(space.nodes[1], 0.0)


def test_apply_clips_instead_of_raising():
    net, ctrl, signals, space = _space(green_bounds=(3.0, 60.0))
    vec = [-10.0] * space.dim  # nonsense proposal: negative greens/offsets
    space.apply(vec)           # must not raise (SignalPlan forbids g <= 0)
    plan = ctrl.plan_for(space.nodes[0])
    assert plan.green_times == (3.0,) * 4 and plan.offset == 0.0


def test_apply_preserves_yellow():
    net, ctrl, signals, space = _space()
    apply_speed_scaled_yellows(signals, braking=2.0)  # long yellows everywhere
    node = space.nodes[0]
    yellow_before = ctrl.plan_for(node).yellow
    assert yellow_before > 1.5
    vec = space.vector()
    vec[0] = 8.0  # change a green time
    space.apply(vec)
    assert ctrl.plan_for(node).yellow == yellow_before
    assert ctrl.plan_for(node).green_times[0] == 8.0


def test_speed_group_scales_from_baseline_and_reroutes():
    net = build_grid_network(width=4, height=4, block=50.0)
    router = ShortestPathRouter(net, seed=0)
    origin, dest = net.node_id[(0, 0)], net.node_id[(3, 3)]
    t0 = router.free_flow_time(origin, dest)  # populate the cache

    edges = [e.id for e in net.edges]
    base = net.edges[0].speed_limit
    space = ParameterSpace(speed_groups={"all": edges}, net=net, router=router,
                           speed_bounds=(0.5, 2.0))
    assert space.dim == 1 and space.labels() == ["speed.all"]

    space.apply([1.5])
    assert math.isclose(net.edges[0].speed_limit, base * 1.5)
    assert math.isclose(router.free_flow_time(origin, dest), t0 / 1.5)
    # Multipliers scale from the baseline, never compound.
    space.apply([1.5])
    assert math.isclose(net.edges[0].speed_limit, base * 1.5)


def test_speed_change_rescales_yellows():
    net, ctrl, signals, space = _space()
    apply_speed_scaled_yellows(signals, braking=4.0)
    node = space.nodes[0]
    yellow_before = ctrl.plan_for(node).yellow
    space2 = ParameterSpace(signals, speed_groups={"all": [e.id for e in net.edges]})
    vec = space2.vector()
    vec[space2.labels().index("speed.all")] = 1.5
    space2.apply(vec)
    # Faster approaches demand a longer clearance interval.
    assert ctrl.plan_for(node).yellow > yellow_before


def test_rejects_empty_and_wrong_size():
    with pytest.raises(ValueError):
        ParameterSpace()
    net, ctrl, signals, space = _space()
    with pytest.raises(ValueError):
        space.apply([5.0])
