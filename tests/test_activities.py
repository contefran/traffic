"""Tests for activity-based per-car daily schedules (traffic_sim.activities)."""

import statistics as st

from traffic_sim import (build_city_grid, assign_zones, edges_by_zone, LandUse,
                         assign_venues, ActivitySchedule, ActivityKind, Car)
from traffic_sim.activities import _edge_mid
import main


def _city(n=14):
    net = build_city_grid(n, n, 150.0, seed=1)
    return net, assign_zones(net, seed=0)


def _scheduled(net, zones, k, **kw):
    cars = [Car(id=i, edge_id=0, s=0.0, v=0.0) for i in range(k)]
    main.assign_homes(cars, zones, seed=0)
    sch = ActivitySchedule(assign_venues(net, zones, seed=0), day_length=1200,
                           seed=0, **kw)
    sch.assign(cars, net, zones)
    return cars, sch


def test_plan_starts_at_work_and_ends_at_home():
    net, zones = _city()
    cars, _ = _scheduled(net, zones, 200)
    for c in cars:
        assert c.plan and c.work is not None
        assert c.plan[0].kind is ActivityKind.WORK     # the day's first trip is to work
        assert c.plan[-1].kind is ActivityKind.HOME    # ...and it ends back home


def test_midnight_state_comes_from_the_schedule():
    net, zones = _city()
    cars, _ = _scheduled(net, zones, 300)
    # everyone is parked (off-road) at 00:00, asleep until a future departure
    assert all(not c.active for c in cars)
    assert all(c.wake_t >= 0.0 for c in cars)
    # most are home overnight; the point is it is *derived* (some are still out)
    assert sum(1 for c in cars if c.edge_id == c.home) > 0.6 * len(cars)


def test_proximity_makes_work_near_home():
    net, zones = _city()
    offices = edges_by_zone(zones)[LandUse.OFFICE]

    def dist(a, b):
        ax, ay = _edge_mid(net, a)
        bx, by = _edge_mid(net, b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    cars, _ = _scheduled(net, zones, 300, work_scale=300.0,
                         long_commute_share=0.0)   # pure Gaussian mechanism
    commute = st.mean(dist(c.home, c.work) for c in cars)
    # what uniform-random office allocation would average, per car
    random_mean = st.mean(st.mean(dist(c.home, o) for o in offices) for c in cars)
    assert commute < 0.6 * random_mean, "proximity should clearly shorten commutes"


def test_long_commute_share_mixes_in_cross_town_jobs():
    net, zones = _city()
    offices = edges_by_zone(zones)[LandUse.OFFICE]

    def dist(a, b):
        ax, ay = _edge_mid(net, a)
        bx, by = _edge_mid(net, b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def mean_commute(share):
        cars, _ = _scheduled(net, zones, 300, work_scale=300.0,
                             long_commute_share=share)
        return st.mean(dist(c.home, c.work) for c in cars)

    near, mixed, uniform = (mean_commute(s) for s in (0.0, 0.3, 1.0))
    # A share of city-wide jobs lengthens the mean commute monotonically,
    # and share=1 approaches what uniform allocation averages.
    assert near < mixed < uniform
    cars, _ = _scheduled(net, zones, 300, work_scale=300.0,
                         long_commute_share=1.0)
    random_mean = st.mean(st.mean(dist(c.home, o) for o in offices) for c in cars)
    assert uniform > 0.8 * random_mean


def test_on_wake_and_on_park_cycle_the_plan():
    net, zones = _city()
    cars, sch = _scheduled(net, zones, 5)
    c = cars[0]
    idx0 = c.plan_idx
    sch.on_wake(c, 0.0)                              # advance to the next leg, head there
    assert c.plan_idx == (idx0 + 1) % len(c.plan)
    leg = c.plan[c.plan_idx]
    assert c.dest_edge == leg.dest_edge
    assert c.dest == net.edges[leg.dest_edge].u     # routes to the street's upstream node
    sch.on_park(c, 100.0)                            # parking sets wake to the next departure
    assert c.wake_t >= 100.0
