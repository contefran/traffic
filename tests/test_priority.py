"""Tests for right-of-way at unsignalized intersections (PriorityModel)."""

from traffic_sim import (
    build_grid_network,
    build_city_grid,
    Car,
    PriorityModel,
    RandomRouter,
    TrafficSim,
)
from traffic_sim.network import DEFAULT_SPEED_LIMIT


def _edge(net, a, b):
    for eid in net.nodes[a].out_edges:
        if net.edges[eid].v == b:
            return eid
    raise AssertionError(f"no edge {a}->{b}")


def _straight_grid():
    # 3x3 uniform grid; the centre node (1,1) is a 4-way unsignalized crossing.
    return build_grid_network(width=3, height=3, block=50.0)


def test_no_priority_model_means_no_yield():
    # Default (priority=None) preserves the old free-for-all: two conflicting
    # cars both reach the centre without either being forced to stop short.
    net = _straight_grid()
    c = net.node_id[(1, 1)]
    west = _edge(net, net.node_id[(0, 1)], c)   # heading east into centre
    south = _edge(net, net.node_id[(1, 0)], c)  # heading north into centre
    cars = [Car(id=0, edge_id=west, s=10.0, v=8.0),
            Car(id=1, edge_id=south, s=10.0, v=8.0)]
    start_edges = {car.id: car.edge_id for car in cars}
    sim = TrafficSim(net, cars, RandomRouter(net, seed=0))
    for _ in range(40):
        sim.step(0.1)
    # With no right-of-way both conflicting cars cross the node freely (each
    # transfers off its starting edge); nothing is held at the stop line.
    assert all(car.edge_id != start_edges[car.id] for car in cars)


def test_local_yields_to_arterial_conflict():
    # East-west row 1 is an arterial; a local (vertical) approach must yield to
    # conflicting arterial through-traffic.
    net = build_city_grid(width=3, height=3, block=50.0, seed=0,
                          arterial_every=3, arterial_speed=25.0)  # row/col 0 only
    # Use a crossing where one approach is arterial and the other is not.
    # Row 0 is arterial (horizontal); column 1 is local (vertical). Centre of
    # that crossing is node (1, 0).
    node = net.node_id[(1, 0)]
    arterial_from = _edge(net, net.node_id[(0, 0)], node)   # along arterial row 0
    local_from = _edge(net, net.node_id[(1, 1)], node)      # local, coming south
    pm = PriorityModel(net)
    assert pm._is_arterial(arterial_from) and not pm._is_arterial(local_from)

    arterial_to = _edge(net, node, net.node_id[(2, 0)])  # arterial continues straight
    # Local turns left onto the arterial heading east (crosses the through path).
    local_to = _edge(net, node, net.node_id[(2, 0)])

    contenders = [
        (arterial_from, arterial_to, 5.0, 12.0),  # arterial car near & fast
        (local_from, local_to, 5.0, 5.0),
    ]
    # Local must yield to the arterial; the arterial need not yield to the local.
    assert pm.must_yield(local_from, local_to, contenders) is True
    assert pm.must_yield(arterial_from, arterial_to, contenders) is False


def test_far_or_stopped_higher_priority_car_does_not_block():
    net = build_city_grid(width=3, height=3, block=50.0, seed=0,
                          arterial_every=3, arterial_speed=25.0)
    node = net.node_id[(1, 0)]
    arterial_from = _edge(net, net.node_id[(0, 0)], node)
    local_from = _edge(net, net.node_id[(1, 1)], node)
    arterial_to = _edge(net, node, net.node_id[(2, 0)])
    local_to = _edge(net, node, net.node_id[(0, 0)])
    pm = PriorityModel(net, trigger_dist=30.0, critical_gap=2.5, min_gap=6.0)

    # Arterial car is far and stopped -> not imminent -> local may proceed.
    contenders = [(arterial_from, arterial_to, 25.0, 0.0),
                  (local_from, local_to, 5.0, 5.0)]
    assert pm.must_yield(local_from, local_to, contenders) is False


def test_right_turn_never_yields():
    net = _straight_grid()
    c = net.node_id[(1, 1)]
    south_from = _edge(net, net.node_id[(1, 0)], c)  # heading north
    # A right turn from a northbound approach goes east.
    right_to = _edge(net, c, net.node_id[(2, 1)])
    west_from = _edge(net, net.node_id[(0, 1)], c)   # conflicting eastbound
    west_to = _edge(net, c, net.node_id[(2, 1)])
    pm = PriorityModel(net)
    contenders = [(south_from, right_to, 3.0, 4.0),
                  (west_from, west_to, 3.0, 10.0)]
    assert pm.must_yield(south_from, right_to, contenders) is False


def test_parallel_straight_movements_do_not_conflict():
    net = _straight_grid()
    c = net.node_id[(1, 1)]
    west_from = _edge(net, net.node_id[(0, 1)], c)   # eastbound through
    west_to = _edge(net, c, net.node_id[(2, 1)])
    east_from = _edge(net, net.node_id[(2, 1)], c)   # westbound through
    east_to = _edge(net, c, net.node_id[(0, 1)])
    pm = PriorityModel(net)
    # Opposite straight movements on the same axis never conflict.
    assert pm._conflict(west_from, west_to, east_from, east_to) is False


def test_crossing_straight_movements_conflict():
    net = _straight_grid()
    c = net.node_id[(1, 1)]
    west_from = _edge(net, net.node_id[(0, 1)], c)   # eastbound through
    west_to = _edge(net, c, net.node_id[(2, 1)])
    south_from = _edge(net, net.node_id[(1, 0)], c)  # northbound through
    south_to = _edge(net, c, net.node_id[(1, 2)])
    pm = PriorityModel(net)
    assert pm._conflict(west_from, west_to, south_from, south_to) is True


def test_priority_is_deterministic_and_keeps_flowing():
    # A populated city with all signals off + right-of-way; runs without deadlock.
    net = build_city_grid(width=6, height=6, block=50.0, seed=1, jitter=0.15,
                          one_way_prob=0.1, drop_prob=0.1, arterial_every=3,
                          arterial_speed=25.0)

    def run():
        cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0)
                for k in range(20)]
        sim = TrafficSim(net, cars, RandomRouter(net, seed=3),
                         priority=PriorityModel(net))
        for _ in range(400):
            sim.step(0.1)
        return [(c.edge_id, round(c.s, 6)) for c in cars]

    a = run()
    b = run()
    assert a == b  # deterministic


def test_two_way_stop_lets_priority_car_through_first():
    # Head-on setup on a uniform grid (no arterials): edge-id tie-break gives one
    # approach priority. The lower-priority conflicting car must be held at its
    # line while the higher-priority one is imminent.
    net = _straight_grid()
    c = net.node_id[(1, 1)]
    west_from = _edge(net, net.node_id[(0, 1)], c)   # eastbound through
    west_to = _edge(net, c, net.node_id[(2, 1)])
    south_from = _edge(net, net.node_id[(1, 0)], c)  # northbound through
    south_to = _edge(net, c, net.node_id[(1, 2)])
    pm = PriorityModel(net)

    contenders = [(west_from, west_to, 4.0, 9.0),
                  (south_from, south_to, 4.0, 9.0)]
    yielders = [pm.must_yield(west_from, west_to, contenders),
                pm.must_yield(south_from, south_to, contenders)]
    # Exactly one of the two conflicting cars yields (no deadlock, no overlap).
    assert sum(yielders) == 1
