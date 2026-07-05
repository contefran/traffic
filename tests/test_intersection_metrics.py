"""Tests for per-intersection throughput and wait metrics."""

from traffic_sim import (
    build_grid_network,
    build_city_grid,
    Car,
    RandomRouter,
    ShortestPathRouter,
    TrafficSim,
    ProtectedPhaseController,
    SignalSystem,
    MetricsCollector,
)


def test_node_crossings_count_transfers_through_nodes():
    net = build_grid_network(4, 4, block=50.0)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(8)]
    m = MetricsCollector()
    sim = TrafficSim(net, cars, RandomRouter(net, seed=0), metrics=m)
    for _ in range(400):
        sim.step(0.1)
    # Every edge->edge transfer crosses exactly one node, so per-node crossings
    # sum to the total crossings.
    total = sum(sm.n_crossings for sm in m.history)
    assert sum(m.node_crossings.values()) == total
    assert total > 0


def test_stopped_car_accumulates_wait_at_its_node():
    # A single car held at a red light banks wait time at the node it faces.
    net = build_city_grid(4, 4, block=120.0, seed=1, arterial_every=2)
    signals = SignalSystem(net, ProtectedPhaseController(green_time=5.0, yellow=1.5))
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    m = MetricsCollector()
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0),
                     signals=signals, metrics=m)
    for _ in range(600):
        sim.step(0.1)
    assert sum(m.node_wait.values()) > 0.0
    # Wait is attributed only to nodes that are the target (v) of some edge.
    for node_id in m.node_wait:
        assert any(e.v == node_id for e in net.edges)


def test_node_mean_wait_and_busiest_nodes():
    net = build_city_grid(6, 6, block=120.0, seed=2, arterial_every=3)
    signals = SignalSystem(net, ProtectedPhaseController(green_time=5.0, yellow=1.5))
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(30)]
    m = MetricsCollector()
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=3), signals=signals, metrics=m)
    for _ in range(500):
        sim.step(0.1)
    busiest = m.busiest_nodes(3)
    assert len(busiest) <= 3
    if busiest:
        waits = [w for _, w, _ in busiest]
        assert waits == sorted(waits, reverse=True)         # worst first
        nid, wait, crossings = busiest[0]
        # mean wait matches the accessor
        assert abs(m.node_mean_wait(nid) - (wait / crossings if crossings else 0.0)) < 1e-9


def test_summary_exposes_intersection_wait():
    net = build_city_grid(5, 5, block=120.0, seed=4, arterial_every=2)
    signals = SignalSystem(net, ProtectedPhaseController(green_time=5.0, yellow=1.5))
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(20)]
    m = MetricsCollector()
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=5), signals=signals, metrics=m)
    for _ in range(400):
        sim.step(0.1)
    s = sim.metrics.summary()
    assert "intersection_wait_s" in s and "mean_wait_per_crossing_s" in s
    assert s["intersection_wait_s"] >= 0.0
