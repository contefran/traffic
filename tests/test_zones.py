"""Tests for the land-use zone overlay (land use lives on **edges**)."""

from collections import Counter

from traffic_sim import (build_city_grid, assign_zones, edges_by_zone,
                         apply_zone_speeds, LandUse, kmh_to_ms)
from traffic_sim.zones import RESIDENTIAL_SPEED


def _ground_edges(net):
    """Edge ids whose both endpoints are ground-level (the zonable streets)."""
    return {e.id for e in net.edges
            if net.nodes[e.u].level == 0 and net.nodes[e.v].level == 0}


def test_every_ground_edge_gets_a_zone():
    net = build_city_grid(8, 8, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    assert set(zones) == _ground_edges(net)
    assert all(isinstance(u, LandUse) for u in zones.values())


def test_both_directions_of_a_street_share_one_use():
    # A two-way street is two directed edges u->v and v->u; they must agree.
    net = build_city_grid(10, 10, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    pairs = {(e.u, e.v): e.id for e in net.edges}
    for e in net.edges:
        rev = pairs.get((e.v, e.u))
        if rev is not None and e.id in zones and rev in zones:
            assert zones[e.id] is zones[rev]


def test_all_land_uses_present_in_polycentric_layout():
    # The default layout scatters several clusters of each use across the map,
    # so every land use should appear (no single central CBD).
    net = build_city_grid(16, 16, 150.0, seed=1)
    zones = assign_zones(net, seed=0)
    counts = Counter(zones.values())
    for use in (LandUse.RESIDENTIAL, LandUse.OFFICE, LandUse.RETAIL):
        assert counts[use] > 0, f"{use} missing from the layout"


def test_sparse_ratio_orders_retail_over_office_over_residential():
    # With no clusters every non-background street is a scatter, so the counts
    # directly expose the per-type scatter probabilities: retail > office >
    # residential (shops are the most ubiquitous, houses the most clustered).
    net = build_city_grid(20, 20, 150.0, seed=2)
    zones = assign_zones(net, seed=0, clusters={})
    counts = Counter(zones.values())
    assert counts[LandUse.RETAIL] > counts[LandUse.OFFICE] > counts[LandUse.RESIDENTIAL]
    # Most of the map is undeveloped background when clusters are switched off.
    assert counts[LandUse.OTHER] > counts[LandUse.RETAIL]


def test_apply_zone_speeds_slows_only_residential_local_streets():
    net = build_city_grid(16, 16, 150.0, seed=1, arterial_every=3,
                          arterial_speed=kmh_to_ms(70), arterial_lanes=2)
    zones = assign_zones(net, seed=0)
    before = {e.id: e.speed_limit for e in net.edges}
    apply_zone_speeds(net, zones)

    slowed = [e for e in net.edges if abs(e.speed_limit - RESIDENTIAL_SPEED) < 1e-6
              and before[e.id] != e.speed_limit]
    assert slowed, "some residential local streets should be slowed"
    for e in slowed:
        assert e.lanes == 1
        assert zones[e.id] is LandUse.RESIDENTIAL
    # Multi-lane arterials are never slowed, even in a residential area.
    for e in net.edges:
        if e.lanes >= 2:
            assert e.speed_limit == before[e.id]


def test_deterministic_and_invertible():
    net = build_city_grid(6, 6, 150.0, seed=3)
    assert assign_zones(net, seed=7) == assign_zones(net, seed=7)
    zones = assign_zones(net, seed=7)
    inv = edges_by_zone(zones)
    assert sum(len(v) for v in inv.values()) == len(zones)
    for use, ids in inv.items():
        assert all(zones[i] is use for i in ids)
