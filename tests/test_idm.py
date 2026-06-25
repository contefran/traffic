"""Tests for the Intelligent Driver Model car-following dynamics."""

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    MetricsCollector,
)


def test_free_road_car_approaches_but_never_exceeds_desired_speed():
    net = build_grid_network(width=4, height=3, block=50.0)
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    v_des = min(net.edges[0].speed_limit, car.max_speed)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=1))
    speeds = []
    for _ in range(60):
        sim.step(0.1)
        speeds.append(car.v)
    assert speeds[-1] > 5.0            # accelerated substantially
    assert max(speeds) <= v_des + 1e-9  # never overshoots the desired speed
    # Monotonic non-decreasing on open road (no obstacle to brake for).
    assert all(b >= a - 1e-9 for a, b in zip(speeds, speeds[1:]))


def test_follower_converges_to_slow_leader_without_overtaking():
    net = build_grid_network(width=4, height=3, block=50.0)
    leader = Car(id=0, edge_id=0, s=25.0, v=3.0, max_speed=3.0)   # capped slow
    follower = Car(id=1, edge_id=0, s=5.0, v=12.0)
    sim = TrafficSim(net, [leader, follower], RandomRouter(net, seed=1))
    for _ in range(30):  # 3s, both still on edge 0
        sim.step(0.1)
        if leader.edge_id == follower.edge_id:
            assert follower.s <= leader.s - follower.length  # never overtakes
    # Follower has slowed toward the leader's pace, not blown past it.
    assert follower.v <= leader.v + 1.0


def test_denser_traffic_has_lower_mean_speed():
    # The fundamental diagram, emergent: more cars -> more interactions -> slower.
    def run(n_cars):
        net = build_grid_network(width=4, height=3, block=50.0)
        cars = [Car(id=i, edge_id=i % len(net.edges), s=5.0, v=0.0)
                for i in range(n_cars)]
        metrics = MetricsCollector()
        sim = TrafficSim(net, cars, RandomRouter(net, seed=3), metrics=metrics)
        for _ in range(300):
            sim.step(0.1)
        return metrics.summary()["avg_speed"]

    sparse = run(3)
    dense = run(24)
    assert dense < sparse
