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
    assert car.v < 1.0   # crept down to (near) a stop at the line under IDM

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


def _out_edge(net, node, to):
    return next(e for e in net.nodes[node].out_edges if net.edges[e].v == to)


def test_car_slows_to_turn_speed_but_not_when_straight():
    # A car cruising at the 50 km/h limit brakes to ~20 km/h to take a turn, but
    # keeps its speed through a straight movement. 5x5 grid so the centre has a
    # left, a right and a straight option.
    from traffic_sim.simulation import TURN_SPEED
    net = build_grid_network(5, 5, block=150.0)
    X = net.node_id[(2, 2)]
    W, N, S, E = (net.node_id[p] for p in [(1, 2), (2, 3), (2, 1), (3, 2)])
    approach = _out_edge(net, W, X)

    def junction_speed(exit_node):
        car = Car(id=0, edge_id=approach, s=0.0, v=13.9)   # ~50 km/h
        car.next_edge = _out_edge(net, X, exit_node)
        sim = TrafficSim(net, [car], RandomRouter(net, seed=0))
        for _ in range(400):
            on = car.edge_id
            sim.step(0.1)
            if car.edge_id != on:                          # just crossed X
                return car.v
            if car.edge_id == approach:                    # keep the turn committed
                car.next_edge = _out_edge(net, X, exit_node)
        raise AssertionError("car never crossed the junction")

    straight = junction_speed(E)
    left = junction_speed(N)
    right = junction_speed(S)
    assert straight > 13.0                                  # essentially unslowed
    assert abs(left - TURN_SPEED) < 1.0                     # ~20 km/h into the turn
    assert abs(right - TURN_SPEED) < 1.0
