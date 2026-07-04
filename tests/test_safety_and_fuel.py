"""Tests for the harsh-braking/crash metric and the fuel/emissions proxy.

The crash/fuel maths is driven directly with hand-crafted speeds through a tiny
sim stub (deterministic, no reliance on emergent dynamics); a couple of real-sim
tests cover the integration behaviour.
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

    def __init__(self, cars, t, net=None, router=None):
        self.cars = cars
        self.t = t
        self.net = net
        self.router = router


_NET = build_grid_network(2, 2, block=50.0)


def _record(metrics, car, times):
    """Feed ``car`` (whose ``v`` the caller mutates) through ``metrics`` at each
    time in ``times``."""
    for t in times:
        metrics.record(_Sim([car], t, _NET))


def test_hard_stop_registers_a_crash_with_correct_decel():
    car = Car(id=0, edge_id=0, s=0.0, v=13.9)
    m = MetricsCollector()
    m.record(_Sim([car], 0.0, _NET))   # seed previous speed (dt = 0)
    car.v = 0.0
    m.record(_Sim([car], 0.1, _NET))   # 13.9 -> 0 over 0.1 s => 139 m/s^2
    assert m.crashes == 1
    assert math.isclose(m.max_decel, 139.0, rel_tol=1e-9)


def test_gentle_deceleration_is_not_a_crash():
    car = Car(id=0, edge_id=0, s=0.0, v=13.9)
    m = MetricsCollector()
    m.record(_Sim([car], 0.0, _NET))
    car.v = 13.5                        # -0.4 m/s over 0.1 s => 4 m/s^2
    m.record(_Sim([car], 0.1, _NET))
    assert m.crashes == 0
    assert math.isclose(m.max_decel, 4.0, rel_tol=1e-9)


def test_implied_deceleration_scales_inversely_with_dt():
    # The same physical stop (13.9 -> 0) implies a larger deceleration when
    # resolved over a smaller step: this is Q14's step-size dependence.
    def max_decel_for(dt):
        car = Car(id=0, edge_id=0, s=0.0, v=13.9)
        m = MetricsCollector()
        m.record(_Sim([car], 0.0, _NET))
        car.v = 0.0
        m.record(_Sim([car], dt, _NET))
        return m.max_decel

    assert max_decel_for(0.1) > max_decel_for(1.0)
    assert math.isclose(max_decel_for(1.0), 13.9, rel_tol=1e-9)


def test_fuel_proxy_charges_idle_and_more_for_acceleration():
    # Idling still burns the base rate.
    idle = Car(id=0, edge_id=0, s=0.0, v=0.0)
    mi = MetricsCollector()
    _record(mi, idle, [0.0, 0.1])
    assert math.isclose(mi.fuel_proxy, FUEL_IDLE * 0.1, rel_tol=1e-9)

    # Accelerating 0 -> 5 m/s over 0.1 s costs the idle + cruise + traction terms.
    acc = Car(id=1, edge_id=0, s=0.0, v=0.0)
    ma = MetricsCollector()
    ma.record(_Sim([acc], 0.0, _NET))
    acc.v = 5.0
    ma.record(_Sim([acc], 0.1, _NET))
    expected = (FUEL_IDLE + FUEL_CRUISE * 5.0 + FUEL_ACCEL * 5.0 * 50.0) * 0.1
    assert math.isclose(ma.fuel_proxy, expected, rel_tol=1e-9)
    assert ma.fuel_proxy > mi.fuel_proxy


def test_lone_free_flowing_car_never_crashes():
    # A single car with no obstacle and uniform speed limits only accelerates and
    # cruises, so it never brakes hard.
    net = build_grid_network(5, 5, block=50.0)
    car = Car(id=0, edge_id=0, s=0.0, v=0.0)
    m = MetricsCollector()
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0), metrics=m)
    for _ in range(300):
        sim.step(0.1)
    assert m.crashes == 0
    assert m.max_decel < m.crash_decel


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
