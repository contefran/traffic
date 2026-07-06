"""Tests for per-car homes and edge-precise (specific-street) arrival."""

from traffic_sim import (
    build_grid_network, build_city_grid, assign_zones, edges_by_zone, LandUse,
    Car, ShortestPathRouter, DemandModel, TrafficSim, ParkingModel,
)
import main


def _edge(net, a, b):
    """The directed edge a -> b."""
    return next(eid for eid in net.nodes[a].out_edges if net.edges[eid].v == b)


def test_assign_homes_covers_every_residential_edge():
    net = build_city_grid(16, 16, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    residential = {eid for eid, u in zones.items() if u is LandUse.RESIDENTIAL}
    # More cars than residential streets, so every street can be someone's home.
    cars = [Car(id=i, edge_id=0, s=0.0, v=0.0) for i in range(len(residential) + 50)]
    main.assign_homes(cars, zones, seed=0)
    assert all(c.home is not None for c in cars)
    assert all(zones[c.home] is LandUse.RESIDENTIAL for c in cars)
    assert residential <= {c.home for c in cars}   # every residential street is a home


def test_residential_destination_returns_own_home():
    net = build_city_grid(16, 16, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    dem = DemandModel(net, zones, seed=0, day_length=400.0)
    res_edges = set(edges_by_zone(zones)[LandUse.RESIDENTIAL])
    home = edges_by_zone(zones)[LandUse.RESIDENTIAL][0]
    office = edges_by_zone(zones)[LandUse.OFFICE][0]

    # Evening from the office: the residential-bound trips must all be *this* home.
    dests = [dem.next_destination(office, t=350.0, home=home) for _ in range(300)]
    residential_dests = [d for d in dests if d in res_edges]
    assert residential_dests, "evening should send some trips to residential"
    assert all(d == home for d in residential_dests)


def test_edge_precise_arrival_parks_on_the_destination_street():
    net = build_grid_network(5, 5, block=200.0)
    router = ShortestPathRouter(net, seed=0, edge_points=True)
    dest_edge = _edge(net, net.node_id[(3, 2)], net.node_id[(4, 2)])
    car = Car(id=0, edge_id=_edge(net, net.node_id[(0, 0)], net.node_id[(1, 0)]),
              s=0.0, v=0.0)
    car.dest_edge = dest_edge                       # a specific street to reach
    car.dest = net.edges[dest_edge].u              # routing target = its upstream node
    car.dest_frac = 0.5                            # address mid-block
    sim = TrafficSim(net, [car], router,
                     parking=ParkingModel(seed=0, default_dwell=(5.0, 5.0)))
    for _ in range(2000):
        sim.step(0.1)
        if not car.active:
            break
    assert not car.active, "car should have parked"
    assert car.edge_id == dest_edge                # on the exact destination street
    length = net.edges[dest_edge].length
    assert abs(car.s - length * 0.5) < length * 0.25   # near the mid-block address
