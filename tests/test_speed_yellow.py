"""Speed-scaled yellow clearance: the dilemma-zone guard.

A uniform yellow sized for slow locals is too short for fast arterials — at
25 m/s a 1.5 s yellow forces a stopping leader far beyond the comfortable
braking rate follower gaps are sized for (the q18 high-speed crash family).
``apply_speed_scaled_yellows`` gives each signalized node
``yellow = max(floor, v_fastest_approach / (2 * braking))`` via per-node
:class:`SignalPlan` overrides.
"""

import pytest

from traffic_sim import (
    build_grid_network,
    FixedTimeController,
    SignalSystem,
    apply_speed_scaled_yellows,
)

FLOOR = 1.5


def _system(fast_speed=None):
    """A 3x3 grid signal system; optionally one 25 m/s approach into the centre."""
    net = build_grid_network(width=3, height=3, block=100.0)
    centre = net.node_id[(1, 1)]
    if fast_speed is not None:
        net.edges[net.nodes[centre].in_edges[0]].speed_limit = fast_speed
    controller = FixedTimeController(green_time=10.0, yellow=FLOOR)
    return net, centre, controller, SignalSystem(net, controller)


def test_fast_approach_gets_kinematic_yellow():
    net, centre, controller, signals = _system(fast_speed=25.0)
    installed = apply_speed_scaled_yellows(signals, braking=4.0)
    # The centre's fastest approach is 25 m/s -> 25 / (2*4) = 3.125 s.
    assert installed[centre] == pytest.approx(3.125)
    assert controller.plan_for(centre).yellow == pytest.approx(3.125)
    # Green times and offset are untouched — only the clearance scales.
    assert controller.plan_for(centre).green_times == \
        controller.default_plan.green_times
    assert signals.yellow_braking == 4.0


def test_floor_binds_on_slow_approaches():
    # With a high braking basis every scaled yellow falls below the floor:
    # no node gets an override, all keep the default plan.
    net, centre, controller, signals = _system(fast_speed=25.0)
    installed = apply_speed_scaled_yellows(signals, braking=100.0)
    assert installed == {}
    assert controller.plan_for(centre) is controller.default_plan


def test_braking_zero_disables():
    net, centre, controller, signals = _system(fast_speed=25.0)
    assert apply_speed_scaled_yellows(signals, braking=0.0) == {}
    assert controller.plans == {}
    assert signals.yellow_braking == 0.0  # recorded, so a dashboard won't re-add


def test_unsignalized_nodes_never_scaled():
    net, centre, controller, signals = _system(fast_speed=25.0)
    signals = SignalSystem(net, controller, unsignalized_nodes={centre})
    installed = apply_speed_scaled_yellows(signals, braking=4.0)
    assert centre not in installed


def test_dashboard_retime_keeps_scaling():
    """The dashboard's yellow slider moves the *floor*; scaled nodes stay long."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from traffic_sim import (Car, Dashboard, MetricsCollector,
                             ShortestPathRouter, TrafficSim)

    net, centre, controller, signals = _system(fast_speed=25.0)
    apply_speed_scaled_yellows(signals, braking=4.0)
    cars = [Car(id=0, edge_id=0, s=5.0, v=0.0)]
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=0),
                     signals=signals, metrics=MetricsCollector())
    dash = Dashboard(net, sim, dt=0.1, day_length=600.0)
    dash.build()
    try:
        dash.sliders["yellow"].set_val(2.0)
        # Slow nodes take the slider's 2.0 s; the fast-approach node keeps its
        # kinematic 3.125 s (scaling re-applied on top of the new floor).
        assert controller.default_plan.yellow == pytest.approx(2.0)
        assert controller.plan_for(centre).yellow == pytest.approx(3.125)
        # Raise the floor past the kinematic value: the floor wins.
        dash.sliders["yellow"].set_val(4.0)
        assert controller.plan_for(centre).yellow == pytest.approx(4.0)
    finally:
        plt.close("all")
