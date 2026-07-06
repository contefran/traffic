"""Tests for the time-of-day demand model and its router integration."""

from collections import Counter

from traffic_sim import (
    build_city_grid, assign_zones, edges_by_zone, LandUse,
    DemandModel, Period, ShortestPathRouter, Car, TrafficSim,
)


def _model(seed=0):
    # 16x16 so the polycentric clusters (radius scales with the grid) actually
    # form; on a tiny grid the districts collapse to single nodes.
    net = build_city_grid(16, 16, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    return net, zones, DemandModel(net, zones, seed=seed, day_length=400.0)


def test_period_splits_the_day_into_four():
    # Day starts at midnight: Night, Morning, Midday, Evening (day_length=400).
    _, _, m = _model()
    assert m.period(0.0) is Period.NIGHT          # 0.00  (midnight)
    assert m.period(150.0) is Period.MORNING      # 0.375 (~09:00)
    assert m.period(250.0) is Period.MIDDAY       # 0.625 (~15:00)
    assert m.period(350.0) is Period.EVENING      # 0.875 (~21:00)
    assert m.period(400.0) is Period.NIGHT        # wraps around to midnight


def test_morning_sends_residents_to_work():
    net, zones, m = _model()
    by = edges_by_zone(zones)
    home = by[LandUse.RESIDENTIAL][0]                  # a residential street
    dests = [zones[m.next_destination(home, t=150.0)] for _ in range(400)]  # morning ~09:00
    counts = Counter(dests)
    # The clear plurality of morning trips from home go to the office.
    assert counts[LandUse.OFFICE] > counts[LandUse.RESIDENTIAL]
    assert counts[LandUse.OFFICE] > 0.5 * len(dests)


def test_evening_sends_workers_home():
    net, zones, m = _model()
    by = edges_by_zone(zones)
    work = by[LandUse.OFFICE][0]                       # an office street
    dests = [zones[m.next_destination(work, t=350.0)] for _ in range(400)]  # evening ~21:00
    counts = Counter(dests)
    assert counts[LandUse.RESIDENTIAL] > counts[LandUse.OFFICE]


def test_never_returns_the_origin():
    net, zones, m = _model()
    home = next(iter(zones))
    assert all(m.next_destination(home, t=t) != home for t in range(0, 400, 7))


def test_router_uses_demand_and_sim_syncs_time():
    net = build_city_grid(16, 16, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    demand = DemandModel(net, zones, seed=0, day_length=400.0)
    router = ShortestPathRouter(net, seed=0, demand=demand)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(8)]
    sim = TrafficSim(net, cars, router)
    sim.step(0.1)
    assert router.now == 0.0                       # synced to the step's start time
    # Every routed car has a real destination *node* (the demand picks a street,
    # the router targets that street's downstream node).
    node_ids = {n.id for n in net.nodes}
    for c in cars:
        if c.dest is not None:
            assert c.dest in node_ids
