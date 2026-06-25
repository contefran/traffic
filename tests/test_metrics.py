import pytest

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    Orientation,
    FixedTimeController,
    SignalSystem,
    MetricsCollector,
)


def test_collector_records_one_entry_per_step():
    net = build_grid_network(width=4, height=3, block=50.0)
    metrics = MetricsCollector()
    sim = TrafficSim(net, [Car(0, 0, 0.0, 0.0)], RandomRouter(net, seed=1), metrics=metrics)
    for _ in range(25):
        sim.step(0.1)
    assert len(metrics.history) == 25
    assert metrics.times[-1] == pytest.approx(2.5)


def test_lone_car_speeds_up_then_queue_is_zero():
    net = build_grid_network(width=4, height=3, block=50.0)
    metrics = MetricsCollector()
    sim = TrafficSim(net, [Car(0, 0, 0.0, 0.0)], RandomRouter(net, seed=1), metrics=metrics)
    for _ in range(50):
        sim.step(0.1)
    assert metrics.mean_speeds[-1] > metrics.mean_speeds[0]
    assert metrics.queue_lengths[-1] == 0  # free-flowing, nothing queued


def test_red_light_builds_a_queue():
    # Cars approach a red horizontal light and pile up.
    net = build_grid_network(width=4, height=3, block=50.0)
    sig = SignalSystem(net, FixedTimeController(green_time=100.0, start=Orientation.VERTICAL))
    cars = [Car(id=i, edge_id=0, s=8.0 + i * 7.0, v=8.0) for i in range(4)]
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, RandomRouter(net, seed=1), signals=sig, metrics=metrics)
    for _ in range(120):  # 12s, all within the red phase
        sim.step(0.1)
    assert metrics.queue_lengths[-1] == 4   # everyone stopped at the line
    assert metrics.summary()["max_queue"] == 4


def test_crossings_counted_as_throughput():
    net = build_grid_network(width=4, height=3, block=50.0)
    length = net.edges[0].length
    metrics = MetricsCollector()
    car = Car(id=0, edge_id=0, s=length - 2.0, v=10.0)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=42), metrics=metrics)
    for _ in range(40):
        sim.step(0.1)
    total = sum(m.n_crossings for m in metrics.history)
    assert total >= 1
    assert metrics.summary()["total_crossings"] == total


def test_occupancy_counts_cars_per_edge():
    net = build_grid_network(width=4, height=3, block=50.0)
    metrics = MetricsCollector()
    cars = [Car(0, 0, 5.0, 0.0), Car(1, 0, 15.0, 0.0), Car(2, 3, 5.0, 0.0)]
    sim = TrafficSim(net, cars, RandomRouter(net, seed=1), metrics=metrics)
    occ = metrics.occupancy(sim)
    assert occ[0] == 2
    assert occ[3] == 1
