"""Tests for the land-use zone overlay."""

from collections import Counter

from traffic_sim import (build_city_grid, assign_zones, nodes_by_zone,
                         apply_zone_speeds, LandUse, kmh_to_ms)
from traffic_sim.zones import RESIDENTIAL_SPEED


def test_every_node_gets_a_zone():
    net = build_city_grid(8, 8, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    assert set(zones) == {n.id for n in net.nodes}
    assert all(isinstance(u, LandUse) for u in zones.values())


def test_centre_is_office_edges_are_residential():
    net = build_city_grid(8, 8, 150.0, seed=1)
    # No retail, so the layout is a clean office core / residential ring.
    zones = assign_zones(net, seed=0, retail_fraction=0.0)
    centre = net.node_id[(4, 4)]
    corner = net.node_id[(0, 0)]
    assert zones[centre] is LandUse.OFFICE
    assert zones[corner] is LandUse.RESIDENTIAL


def test_retail_fraction_scatters_shops():
    net = build_city_grid(10, 10, 150.0, seed=2)
    zones = assign_zones(net, seed=0, retail_fraction=0.3)
    counts = Counter(zones.values())
    assert counts[LandUse.RETAIL] > 0
    # Roughly the requested fraction (loose bounds for randomness).
    frac = counts[LandUse.RETAIL] / len(zones)
    assert 0.15 < frac < 0.45


def test_apply_zone_speeds_slows_only_residential_local_streets():
    net = build_city_grid(8, 8, 150.0, seed=1, arterial_every=3,
                          arterial_speed=kmh_to_ms(70), arterial_lanes=2)
    zones = assign_zones(net, seed=0, retail_fraction=0.0)
    before = {e.id: e.speed_limit for e in net.edges}
    apply_zone_speeds(net, zones)

    slowed = [e for e in net.edges if abs(e.speed_limit - RESIDENTIAL_SPEED) < 1e-6
              and before[e.id] != e.speed_limit]
    assert slowed, "some residential local streets should be slowed"
    for e in slowed:
        assert e.lanes == 1
        assert zones[e.u] is LandUse.RESIDENTIAL and zones[e.v] is LandUse.RESIDENTIAL
    # Multi-lane arterials are never slowed, even in a residential area.
    for e in net.edges:
        if e.lanes >= 2:
            assert e.speed_limit == before[e.id]


def test_deterministic_and_invertible():
    net = build_city_grid(6, 6, 150.0, seed=3)
    assert assign_zones(net, seed=7) == assign_zones(net, seed=7)
    zones = assign_zones(net, seed=7)
    inv = nodes_by_zone(zones)
    assert sum(len(v) for v in inv.values()) == len(zones)
    for use, ids in inv.items():
        assert all(zones[i] is use for i in ids)
