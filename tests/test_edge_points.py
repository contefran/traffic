"""Tests for edge-point destinations (a car parks mid-block, not at the node)."""

from traffic_sim import (
    build_grid_network, build_city_grid, Car, ShortestPathRouter,
    TrafficSim, ParkingModel, MetricsCollector,
)


def _edge(net, a, b):
    return next(eid for eid in net.nodes[a].out_edges if net.edges[eid].v == b)


def test_car_parks_mid_block_at_edge_point():
    net = build_grid_network(4, 4, block=200.0)
    dest = net.node_id[(2, 1)]
    start, nbr = net.node_id[(0, 1)], net.node_id[(1, 1)]
    car = Car(id=0, edge_id=_edge(net, start, nbr), s=0.0, v=0.0,
              dest=dest, dest_frac=0.5)          # a point halfway along the final edge
    parking = ParkingModel(seed=0, default_dwell=(5.0, 5.0))
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0), parking=parking)
    for _ in range(600):
        sim.step(0.1)
        if not car.active:
            break
    assert not car.active, "car should have parked"
    final = net.edges[car.edge_id]
    assert final.v == dest                        # on the final approach edge
    # It stopped near the mid-block point, well short of the intersection.
    assert car.s < final.length * 0.7
    assert abs(car.s - final.length * 0.5) < final.length * 0.25


def test_edge_points_router_sets_fraction():
    net = build_city_grid(6, 6, 150.0, seed=1)
    router = ShortestPathRouter(net, seed=0, edge_points=True)
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    router.assign_destination(car, avoid=net.edges[0].v)
    assert 0.0 < car.dest_frac < 1.0              # a mid-block fraction

    plain = ShortestPathRouter(net, seed=0)       # edge_points off -> junctions
    car2 = Car(id=1, edge_id=0, s=0.0, v=0.0)
    plain.assign_destination(car2, avoid=net.edges[0].v)
    assert car2.dest_frac == 1.0


def test_delay_stays_nonnegative_with_edge_points():
    net = build_city_grid(6, 6, 150.0, seed=1, arterial_every=3)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(20)]
    m = MetricsCollector()
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=1, edge_points=True),
                     parking=ParkingModel(seed=0, default_dwell=(10.0, 20.0)), metrics=m)
    for _ in range(1500):
        sim.step(0.1)
    assert m.trips
    assert all(tp.delay >= 0.0 for tp in m.trips)
