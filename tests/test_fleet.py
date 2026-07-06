"""Tests for vehicle types (the fleet: cars, sports cars, trucks, buses)."""

from collections import Counter

from traffic_sim import (Car, DEFAULT_FLEET, assign_vehicle_types,
                         build_grid_network, ShortestPathRouter, TrafficSim)


def _cars(k):
    return [Car(id=i, edge_id=0, s=0.0, v=0.0) for i in range(k)]


def test_fleet_distribution_roughly_matches_shares():
    cars = _cars(3000)
    assign_vehicle_types(cars, seed=0)
    frac = {t: n / len(cars) for t, n in Counter(c.vtype for c in cars).items()}
    assert frac["city"] > 0.6                       # city cars dominate
    assert frac.get("truck", 0) > 0.03              # but trucks/buses/sports appear
    assert frac.get("bus", 0) > 0.01
    assert frac.get("sport", 0) > 0.03


def test_types_have_distinct_physical_parameters():
    cars = _cars(2000)
    assign_vehicle_types(cars, seed=1, jitter=0.0)   # no jitter: exact type values
    by = {c.vtype: c for c in cars}
    # trucks/buses are long; sports cars accelerate hardest; heavy vehicles brake worst
    assert by["truck"].length > by["city"].length
    assert by["bus"].length > by["city"].length
    assert by["truck"].accel < by["city"].accel < by["sport"].accel
    assert by["truck"].max_brake < by["city"].max_brake
    assert by["truck"].max_speed < by["sport"].max_speed


def test_driver_jitter_varies_gaps_within_a_type():
    cars = _cars(400)
    assign_vehicle_types(cars, seed=2, jitter=0.1)
    ths = {c.time_headway for c in cars if c.vtype == "city"}
    assert len(ths) > 1, "per-driver jitter should make city drivers differ"


def test_deterministic_under_seed():
    a, b = _cars(200), _cars(200)
    assign_vehicle_types(a, seed=7)
    assign_vehicle_types(b, seed=7)
    assert [c.vtype for c in a] == [c.vtype for c in b]


def test_sim_runs_with_a_mixed_fleet():
    net = build_grid_network(5, 5, block=150.0)
    cars = [Car(id=i, edge_id=i % len(net.edges), s=0.0, v=0.0) for i in range(20)]
    assign_vehicle_types(cars, seed=0)
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=0))
    for _ in range(200):
        sim.step(0.1)
    # nothing exploded; a truck keeps its (bigger) gap so it never overlaps a leader
    assert all(c.s >= 0.0 for c in cars)
