"""Tests for trip-level metrics: travel time, delay, stops, and fairness."""

import math

from traffic_sim import (
    build_grid_network,
    build_city_grid,
    Car,
    RandomRouter,
    ShortestPathRouter,
    TrafficSim,
    MetricsCollector,
)


def _edge_between(net, a, b):
    for eid in net.nodes[a].out_edges:
        if net.edges[eid].v == b:
            return eid
    raise AssertionError(f"no edge {a}->{b}")


def test_free_flow_time_matches_cost_to_go():
    net = build_grid_network(width=4, height=4, block=50.0)
    router = ShortestPathRouter(net, seed=0)
    dest = net.node_id[(3, 3)]
    origin = net.node_id[(0, 0)]
    # Uniform grid: 6 steps of length/speed each.
    step_cost = 50.0 / net.edges[0].speed_limit
    assert math.isclose(router.free_flow_time(origin, dest), 6 * step_cost, rel_tol=1e-9)
    # Reaching the destination itself costs nothing.
    assert router.free_flow_time(dest, dest) == 0.0


def test_trips_are_recorded_and_delay_is_nonnegative():
    net = build_grid_network(width=5, height=5, block=50.0)
    router = ShortestPathRouter(net, seed=1)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(8)]
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, router, metrics=metrics)
    for _ in range(1500):
        sim.step(0.1)

    assert metrics.trips, "expected some completed trips"
    for tp in metrics.trips:
        # Actual time can never beat the free-flow ideal (small tolerance for the
        # origin approximation at the moment a destination is assigned).
        assert tp.travel_time >= tp.free_flow_time - 1e-6
        assert tp.delay >= -1e-6
        assert tp.stopped_time >= 0.0 and tp.stops >= 0


def test_delay_is_positive_when_signals_force_stops():
    # A single car crossing a signalized grid must wait at reds, so at least one
    # completed trip should show a real delay and a stop.
    net = build_city_grid(width=5, height=5, block=60.0, seed=3, arterial_every=2)
    from traffic_sim import SignalSystem, ProtectedPhaseController
    router = ShortestPathRouter(net, seed=2)
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    metrics = MetricsCollector()
    sim = TrafficSim(net, [car], router,
                     signals=SignalSystem(net, ProtectedPhaseController(green_time=5.0)),
                     metrics=metrics)
    for _ in range(3000):
        sim.step(0.1)

    assert metrics.trips
    assert any(tp.delay > 0 for tp in metrics.trips)
    assert any(tp.stops > 0 for tp in metrics.trips)


def test_summary_reports_trip_and_fairness_fields():
    net = build_grid_network(width=5, height=5, block=50.0)
    router = ShortestPathRouter(net, seed=1)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(8)]
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, router, metrics=metrics)
    for _ in range(1500):
        sim.step(0.1)

    s = sim.metrics.summary()
    for key in ("trips_completed", "mean_delay_s", "median_delay_s",
                "p90_delay_s", "p95_delay_s", "mean_stops_per_trip",
                "delay_p95_over_mean"):
        assert key in s
    assert s["trips_completed"] == len(metrics.trips)
    assert s["p95_delay_s"] >= s["median_delay_s"]


def test_wandering_router_produces_no_trips():
    # RandomRouter never assigns a destination, so there are no trips to measure.
    net = build_grid_network(width=5, height=5, block=50.0)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(8)]
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, RandomRouter(net, seed=0), metrics=metrics)
    for _ in range(500):
        sim.step(0.1)

    assert metrics.trips == []
    assert "trips_completed" not in sim.metrics.summary()
