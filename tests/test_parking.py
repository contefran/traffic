"""Tests for park-and-dwell (cars stop at destinations, then set off again)."""

from traffic_sim import (
    build_grid_network, build_city_grid, Car, ShortestPathRouter, TrafficSim,
    ParkingModel, MetricsCollector,
)


def _edge(net, a, b):
    return next(eid for eid in net.nodes[a].out_edges if net.edges[eid].v == b)


def test_car_parks_at_destination_then_wakes():
    net = build_grid_network(4, 4, block=150.0)
    dest = net.node_id[(2, 1)]
    start, nbr = net.node_id[(0, 1)], net.node_id[(1, 1)]
    car = Car(id=0, edge_id=_edge(net, start, nbr), s=0.0, v=0.0, dest=dest)
    parking = ParkingModel(seed=0, default_dwell=(3.0, 3.0))   # fixed 3 s dwell
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0), parking=parking)

    for _ in range(500):
        sim.step(0.1)
        if not car.active:
            break
    assert not car.active, "car should have parked at its destination"
    assert net.edges[car.edge_id].v == dest        # parked on the edge into dest

    parked_edge = car.edge_id
    for _ in range(120):                            # run past the 3 s dwell
        sim.step(0.1)
        if car.active:
            break
    assert car.active, "car should have woken after its dwell"


def test_parking_reduces_active_cars_and_keeps_trips_clean():
    net = build_city_grid(6, 6, 150.0, seed=1, arterial_every=3)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(30)]
    parking = ParkingModel(seed=0, default_dwell=(20.0, 40.0))
    m = MetricsCollector()
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=1), parking=parking, metrics=m)
    min_active = len(cars)
    for _ in range(1500):
        sim.step(0.1)
        min_active = min(min_active, sum(1 for c in cars if c.active))

    assert min_active < len(cars), "some cars should be parked at some point"
    assert m.trips, "trips should still be recorded"
    # Travel times must not include the 20-40 s dwell (trip ends at arrival).
    assert max(tp.travel_time for tp in m.trips) < 1000.0
    for tp in m.trips:
        assert tp.delay >= -1e-6


def test_no_parking_leaves_everyone_active():
    net = build_grid_network(4, 4, block=150.0)
    cars = [Car(id=k, edge_id=k, s=0.0, v=0.0) for k in range(5)]
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=0))   # parking=None
    for _ in range(300):
        sim.step(0.1)
    assert all(c.active for c in cars)
