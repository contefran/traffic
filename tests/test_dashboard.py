"""Tests for the interactive dashboard (headless: widgets poked directly)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from traffic_sim import (
    build_grid_network,
    Car,
    Dashboard,
    FixedTimeController,
    MetricsCollector,
    ShortestPathRouter,
    SignalSystem,
    TrafficSim,
    assign_vehicle_types,
)


def _make(n_cars=6):
    """A small signalized sim with metrics, plus its dashboard (built)."""
    net = build_grid_network(width=4, height=4, block=100.0)
    cars = [Car(id=i, edge_id=i % len(net.edges), s=5.0, v=0.0)
            for i in range(n_cars)]
    assign_vehicle_types(cars, seed=0)
    signals = SignalSystem(net, FixedTimeController(green_time=6.0, yellow=1.5))
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=0),
                     signals=signals, metrics=MetricsCollector())
    dash = Dashboard(net, sim, dt=0.1, day_length=600.0)
    dash.build()
    return net, sim, dash


def teardown_function(_fn):
    plt.close("all")


def test_speed_knob_scales_from_baseline_without_compounding():
    net, _sim, dash = _make()
    base = {e.id: e.speed_limit for e in net.edges}
    dash.sliders["local"].set_val(0.5)
    assert all(net.edges[eid].speed_limit == pytest.approx(0.5 * base[eid])
               for eid in dash._local_edges)
    dash.sliders["local"].set_val(0.5)      # drag again: still 0.5x, not 0.25x
    dash.sliders["local"].set_val(1.0)      # back to the original limits
    assert all(net.edges[eid].speed_limit == pytest.approx(base[eid])
               for eid in dash._local_edges)


def test_following_knob_scales_gaps_and_keeps_driver_jitter():
    _net, sim, dash = _make(n_cars=20)
    base = {c.id: (c.time_headway, c.s0) for c in sim.cars}
    dash.sliders["follow"].set_val(2.0)
    for c in sim.cars:                      # each driver doubled from *their* value
        assert c.time_headway == pytest.approx(2.0 * base[c.id][0])
        assert c.s0 == pytest.approx(2.0 * base[c.id][1])


def test_signal_knobs_retime_every_plan():
    _net, sim, dash = _make()
    dash.sliders["green"].set_val(8.0)
    plan = sim.signals.controller.default_plan
    assert plan.green_times == (8.0, 8.0)   # 2-phase fixed-time controller
    assert plan.yellow == pytest.approx(1.5)  # yellow slider untouched
    dash.sliders["yellow"].set_val(3.0)
    assert sim.signals.controller.default_plan.yellow == pytest.approx(3.0)


def test_update_advances_sim_and_fills_panel():
    _net, sim, dash = _make()
    dash.sliders["speed"].set_val(3)        # 3 sim steps per frame
    t0 = sim.t
    dash._update(0)
    assert sim.t == pytest.approx(t0 + 3 * 0.1)
    assert "speed now" in dash._panel_text()


def test_reset_button_starts_a_fresh_metrics_window():
    _net, sim, dash = _make()
    for _ in range(5):
        dash._update(0)
    assert len(sim.metrics.history) > 0
    dash._on_reset()
    assert sim.metrics.history == [] and sim.metrics.trips == []
    dash._update(0)                          # collection resumes cleanly
    assert len(sim.metrics.history) == 1
