"""Tests for the per-day demand intensity (ActivitySchedule intensity=)."""

import math

import pytest

from ml.dataset import generate
from traffic_sim import (ActivitySchedule, assign_venues, assign_zones,
                         build_grid_network, Car, ShortestPathRouter,
                         TrafficSim)
from main import assign_homes


def _city(n_cars=40, seed=3):
    net = build_grid_network(width=6, height=6, block=60.0)
    zones = assign_zones(net, seed=seed)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0)
            for k in range(n_cars)]
    assign_homes(cars, zones, seed=seed)
    venues = assign_venues(net, zones, seed=seed)
    return net, zones, cars, venues


def test_full_intensity_reproduces_old_plans_exactly():
    # intensity=1.0 must not consume any extra RNG draws, so the plans are
    # bit-for-bit those of a schedule built without the parameter.
    net, zones, cars_a, venues = _city()
    _, _, cars_b, _ = _city()
    ActivitySchedule(venues, day_length=600.0, seed=7).assign(cars_a, net, zones)
    ActivitySchedule(venues, day_length=600.0, seed=7,
                     intensity=1.0).assign(cars_b, net, zones)
    for a, b in zip(cars_a, cars_b):
        assert a.work == b.work and a.wake_t == b.wake_t
        assert [(l.depart, l.dest_edge) for l in a.plan] == \
               [(l.depart, l.dest_edge) for l in b.plan]


def test_zero_intensity_keeps_everyone_home():
    net, zones, cars, venues = _city()
    ActivitySchedule(venues, day_length=600.0, seed=7,
                     intensity=0.0).assign(cars, net, zones)
    for car in cars:
        assert car.plan == [] and not car.active
        assert car.edge_id == car.home and math.isinf(car.wake_t)


def test_partial_intensity_mixes_and_sleepers_never_drive():
    net, zones, cars, venues = _city(n_cars=60)
    sched = ActivitySchedule(venues, day_length=600.0, seed=7, intensity=0.5)
    sched.assign(cars, net, zones)
    sleepers = [c for c in cars if not c.plan]
    commuters = [c for c in cars if c.plan]
    assert sleepers and commuters          # genuinely mixed at 0.5

    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=1), schedule=sched)
    for _ in range(300):
        sim.step(0.1)
    assert all(not c.active for c in sleepers)   # a sleeper never wakes


def test_intensity_validation():
    net, zones, cars, venues = _city()
    with pytest.raises(ValueError):
        ActivitySchedule(venues, intensity=1.5)


def test_dataset_records_per_day_intensity(tmp_path):
    tiny = dict(width=8, height=8, day_length=30.0,
                grade=False, ring=False, roundabouts=0,
                parking=False, demand=False)
    manifest = generate(tmp_path, loads=(10,), seeds=(1, 2), bin_s=5.0,
                        workers=1, intensity_range=(0.5, 1.0), **tiny)
    vals = [r["intensity"] for r in manifest["runs"]]
    assert all(0.5 <= v <= 1.0 for v in vals)
    assert vals[0] != vals[1]              # different days, different draws
    # And the default range keeps full demand.
    m2 = generate(tmp_path / "flat", loads=(10,), seeds=(1,), bin_s=5.0,
                  workers=1, **tiny)
    assert m2["runs"][0]["intensity"] == 1.0
