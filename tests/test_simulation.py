import pytest

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    Orientation,
    FixedTimeController,
    SignalSystem,
)


def make_sim(cars, seed=0):
    net = build_grid_network(width=4, height=3, block=50.0)
    return net, TrafficSim(net, cars, RandomRouter(net, seed=seed))


def test_lone_car_accelerates_toward_speed_limit():
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    net, sim = make_sim([car])
    v_des = min(net.edges[0].speed_limit, car.max_speed)
    for _ in range(50):
        sim.step(0.1)
    assert car.v > 0.0
    assert car.v <= v_des + 1e-9


def test_follower_keeps_gap_and_never_passes_leader():
    leader = Car(id=0, edge_id=0, s=30.0, v=0.0)
    follower = Car(id=1, edge_id=0, s=5.0, v=10.0)
    net, sim = make_sim([leader, follower])
    for _ in range(100):
        sim.step(0.1)
        if leader.edge_id == follower.edge_id:
            assert follower.s <= leader.s - follower.length
        # The follower must keep at least the standstill gap once stopped.
    assert follower.v >= 0.0


def test_car_crosses_intersection_instead_of_stopping():
    # Start near the end of edge 0; it must move onto a new edge, not freeze.
    net = build_grid_network(width=4, height=3, block=50.0)
    edge0_len = net.edges[0].length
    car = Car(id=0, edge_id=0, s=edge0_len - 1.0, v=10.0)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=42))
    for _ in range(50):
        sim.step(0.1)
    assert car.edge_id != 0  # transferred onto a downstream edge


def test_car_stops_at_red_light():
    # Edge 0 is horizontal ((0,0)->(1,0)); start=VERTICAL keeps it red.
    net = build_grid_network(width=4, height=3, block=50.0)
    length = net.edges[0].length
    sig = SignalSystem(net, FixedTimeController(green_time=100.0, start=Orientation.VERTICAL))
    car = Car(id=0, edge_id=0, s=length - 15.0, v=10.0)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=1), signals=sig)
    for _ in range(100):  # 10s, still well within the red phase
        sim.step(0.1)
    assert car.edge_id == 0                 # did not cross the intersection
    assert car.s <= length + 1e-9           # did not pass the stop line
    # Stops just short of the line (within the standstill gap s0), not on it.
    assert length - (car.s0 + 1.0) <= car.s <= length
    assert car.v == pytest.approx(0.0, abs=1e-6)


def test_car_waits_then_crosses_on_green():
    net = build_grid_network(width=4, height=3, block=50.0)
    length = net.edges[0].length
    # Horizontal red for t in [0,5), green for t in [5,10).
    sig = SignalSystem(net, FixedTimeController(green_time=5.0, start=Orientation.VERTICAL))
    car = Car(id=0, edge_id=0, s=length - 10.0, v=10.0)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=1), signals=sig)

    for _ in range(30):   # t -> 3s, still red
        sim.step(0.1)
    assert car.edge_id == 0
    assert car.v == pytest.approx(0.0, abs=1e-6)

    for _ in range(60):   # t -> 9s, light has gone green
        sim.step(0.1)
    assert car.edge_id != 0  # crossed once the light allowed it


def test_dead_end_stops_car():
    # A 1x2 grid: node 0 -> node 1 via edge 0; node 1's only out-edge is the
    # U-turn back to 0, so with U-turns allowed the car keeps moving; here we
    # verify a clamp happens when there is genuinely no out-edge.
    net = build_grid_network(width=1, height=2, block=50.0)
    # Strip node 1's out-edges to simulate a true dead-end.
    net.nodes[1].out_edges.clear()
    car = Car(id=0, edge_id=0, s=net.edges[0].length - 0.5, v=10.0)
    sim = TrafficSim(net, [car], RandomRouter(net, seed=0))
    for _ in range(20):
        sim.step(0.1)
    assert car.edge_id == 0
    assert car.s == net.edges[0].length
    assert car.v == 0.0
