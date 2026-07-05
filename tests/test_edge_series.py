"""Tests for the per-edge spatial time series (flow-model substrate)."""

import math

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    MetricsCollector,
)


def test_edge_series_off_by_default():
    net = build_grid_network(3, 3, block=50.0)
    cars = [Car(id=0, edge_id=0, s=0.0, v=0.0)]
    m = MetricsCollector()
    sim = TrafficSim(net, cars, RandomRouter(net, seed=0), metrics=m)
    for _ in range(20):
        sim.step(0.1)
    assert m.edge_history == []
    assert m.edge_count_series(0) == []


def test_edge_series_records_counts_and_speeds():
    net = build_grid_network(3, 3, block=50.0)
    # Two cars parked on edge 0 (v=0) so counts/speeds are predictable at step 1.
    a = Car(id=0, edge_id=0, s=10.0, v=0.0)
    b = Car(id=1, edge_id=0, s=5.0, v=0.0)
    m = MetricsCollector(record_edges=True)
    sim = TrafficSim(net, [a, b], RandomRouter(net, seed=0), metrics=m)
    sim.step(0.1)
    assert len(m.edge_history) == 1
    assert m.edge_count_series(0)[0] == 2
    # Both were stopped this step (blocked by nothing but v started at 0).
    assert m.edge_speed_series(0)[0] >= 0.0
    # An empty edge reads back as zero.
    assert m.edge_count_series(1)[0] == 0
    assert m.edge_speed_series(1)[0] == 0.0


def test_edge_series_length_matches_history():
    net = build_grid_network(4, 4, block=50.0)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(6)]
    m = MetricsCollector(record_edges=True)
    sim = TrafficSim(net, cars, RandomRouter(net, seed=1), metrics=m)
    for _ in range(30):
        sim.step(0.1)
    assert len(m.edge_history) == len(m.history) == 30
    assert len(m.edge_speed_series(0)) == 30


def test_fundamental_samples_obey_q_equals_k_v():
    net = build_grid_network(4, 4, block=50.0)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(10)]
    m = MetricsCollector(record_edges=True)
    sim = TrafficSim(net, cars, RandomRouter(net, seed=2), metrics=m)
    for _ in range(50):
        sim.step(0.1)
    samples = m.fundamental_samples(net)
    assert samples, "expected occupied-edge samples"
    for k, q, v in samples:
        assert k > 0 and v >= 0
        assert math.isclose(q, k * v, rel_tol=1e-9)
