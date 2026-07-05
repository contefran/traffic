"""Tests for the collision metric (bounded-deceleration model) and fuel proxy.

Under the bounded-deceleration model a car can never brake harder than its
physical ``max_brake``; a *crash* is a car that still could not stop in time
(would overlap its leader or cross a red line). ``TrafficSim`` counts these and
the collector mirrors the count. The fuel proxy is collector-side and is driven
directly with hand-crafted speeds through a tiny sim stub.
"""

import math

from traffic_sim import (
    build_grid_network,
    Car,
    ShortestPathRouter,
    TrafficSim,
    MetricsCollector,
)
from traffic_sim.metrics import FUEL_IDLE, FUEL_CRUISE, FUEL_ACCEL


class _Sim:
    """Minimal stand-in exposing just what ``MetricsCollector.record`` reads."""

    def __init__(self, cars, t, net=None, router=None, crashes=0):
        self.cars = cars
        self.t = t
        self.net = net
        self.router = router
        self.crashes = crashes


_NET = build_grid_network(2, 2, block=50.0)


def test_deceleration_is_capped_at_max_brake():
    # A fast car meeting a nearby stationary leader brakes hard, but never harder
    # than its physical limit -- and if it stops in the available room, no crash.
    net = build_grid_network(3, 3, block=50.0)
    leader = Car(id=0, edge_id=0, s=40.0, v=0.0)
    follower = Car(id=1, edge_id=0, s=0.0, v=13.9)
    m = MetricsCollector()
    sim = TrafficSim(net, [leader, follower], ShortestPathRouter(net, seed=0), metrics=m)
    for _ in range(60):
        sim.step(0.1)
    assert sim.crashes == 0                      # 40 m of room is enough at 9 m/s^2
    assert m.max_decel <= follower.max_brake + 1e-6


def test_unavoidable_overlap_is_counted_as_a_crash():
    # A fast car placed far too close to a stationary leader cannot stop even at
    # max braking, so it collides -- counted, but still not overlapping visually.
    net = build_grid_network(3, 3, block=50.0)
    leader = Car(id=0, edge_id=0, s=5.0, v=0.0)      # ~bumper to bumper
    follower = Car(id=1, edge_id=0, s=0.0, v=13.9)
    m = MetricsCollector()
    sim = TrafficSim(net, [leader, follower], ShortestPathRouter(net, seed=0), metrics=m)
    sim.step(0.1)
    assert sim.crashes >= 1
    assert m.crashes == sim.crashes              # collector mirrors the sim count
    # No visual overlap: the follower stays behind the leader.
    assert follower.s <= leader.s - follower.length + 1e-6


def test_lone_free_flowing_car_never_crashes():
    net = build_grid_network(5, 5, block=50.0)
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    m = MetricsCollector()
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0), metrics=m)
    for _ in range(300):
        sim.step(0.1)
    assert sim.crashes == 0
    assert m.max_decel <= car.max_brake + 1e-6


def test_fuel_proxy_charges_idle_and_more_for_acceleration():
    # Idling still burns the base rate.
    idle = Car(id=0, edge_id=0, s=0.0, v=0.0)
    mi = MetricsCollector()
    mi.record(_Sim([idle], 0.0, _NET))
    mi.record(_Sim([idle], 0.1, _NET))
    assert math.isclose(mi.fuel_proxy, FUEL_IDLE * 0.1, rel_tol=1e-9)

    # Accelerating 0 -> 5 m/s over 0.1 s costs idle + cruise + traction terms.
    acc = Car(id=1, edge_id=0, s=0.0, v=0.0)
    ma = MetricsCollector()
    ma.record(_Sim([acc], 0.0, _NET))
    acc.v = 5.0
    ma.record(_Sim([acc], 0.1, _NET))
    expected = (FUEL_IDLE + FUEL_CRUISE * 5.0 + FUEL_ACCEL * 5.0 * 50.0) * 0.1
    assert math.isclose(ma.fuel_proxy, expected, rel_tol=1e-9)
    assert ma.fuel_proxy > mi.fuel_proxy


def test_summary_exposes_safety_and_fuel_fields():
    net = build_grid_network(4, 4, block=50.0)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(5)]
    m = MetricsCollector()
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=0), metrics=m)
    for _ in range(50):
        sim.step(0.1)
    s = sim.metrics.summary()
    for key in ("crashes", "max_decel", "fuel_proxy"):
        assert key in s
    assert isinstance(s["crashes"], int)
    assert s["fuel_proxy"] > 0.0
