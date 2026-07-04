import math

from traffic_sim import build_grid_network, build_city_grid
from traffic_sim.network import DEFAULT_SPEED_LIMIT
from traffic_sim.signals import edge_orientation, Orientation, SignalSystem, FixedTimeController


def test_plain_city_grid_matches_regular_grid():
    # With no heterogeneity, it should reproduce the regular grid.
    city = build_city_grid(4, 3, block=50.0)
    grid = build_grid_network(4, 3, block=50.0)
    assert len(city.nodes) == len(grid.nodes)
    assert len(city.edges) == len(grid.edges)
    for e in city.edges:
        assert math.isclose(e.length, 50.0)


def test_preserves_grid_indices_for_signals():
    # i/j must survive jitter so the H/V signal model still classifies edges.
    city = build_city_grid(5, 5, block=50.0, seed=1, jitter=0.3)
    assert city.node_id[(0, 0)] == 0
    sig = SignalSystem(city, FixedTimeController())
    # Interior nodes still mix both orientations -> signalized.
    assert sig.is_signalized(city.node_id[(2, 2)]) is True
    horiz = [e for e in city.edges if edge_orientation(city, e.id) is Orientation.HORIZONTAL]
    vert = [e for e in city.edges if edge_orientation(city, e.id) is Orientation.VERTICAL]
    assert horiz and vert


def test_jitter_moves_positions_but_not_indices():
    city = build_city_grid(4, 4, block=50.0, seed=2, jitter=0.25)
    moved = any(not math.isclose(n.x, n.i * 50.0) or not math.isclose(n.y, n.j * 50.0)
               for n in city.nodes)
    assert moved


def test_one_way_reduces_edge_count():
    two_way = build_city_grid(6, 6, block=50.0, seed=3, one_way_prob=0.0)
    mostly_one_way = build_city_grid(6, 6, block=50.0, seed=3, one_way_prob=1.0)
    # One-way connections roughly halve the edge count; the repair pass may add
    # a few two-way edges back to remove sinks/sources, so the result is fewer
    # than the fully two-way grid but no less than half of it.
    assert len(two_way.edges) // 2 <= len(mostly_one_way.edges) < len(two_way.edges)


def test_drop_prob_removes_connections():
    full = build_city_grid(6, 6, block=50.0, seed=3)
    dropped = build_city_grid(6, 6, block=50.0, seed=3, drop_prob=0.3)
    assert len(dropped.edges) < len(full.edges)


def test_every_node_has_in_and_out_edge():
    # Even at high drop/one-way rates the repair pass guarantees no node is a
    # trap (no out-edge) or unreachable (no in-edge).
    city = build_city_grid(8, 8, block=50.0, seed=1, jitter=0.2,
                          one_way_prob=0.3, drop_prob=0.25, arterial_every=3)
    assert all(n.out_edges and n.in_edges for n in city.nodes)


def test_no_dead_ends_every_node_has_two_distinct_exits():
    # The no-dead-end repair guarantees every node can leave toward >= 2 distinct
    # nodes, so an arrival is never forced into a U-turn. Check across seeds and
    # aggressive drop/one-way rates.
    for seed in range(6):
        city = build_city_grid(8, 8, block=50.0, seed=seed, jitter=0.2,
                              one_way_prob=0.3, drop_prob=0.3, arterial_every=3)
        for n in city.nodes:
            exits = {city.edges[eid].v for eid in n.out_edges}
            assert len(exits) >= 2, f"node {n.id} (seed {seed}) is a dead-end"


def test_arterials_are_never_dropped():
    # With drop_prob=1.0 only arterial links survive (they are never dropped),
    # plus whatever the repair pass restores.
    city = build_city_grid(6, 6, block=50.0, seed=3, drop_prob=1.0,
                          arterial_every=2, arterial_speed=25.0)
    assert any(e.speed_limit == 25.0 for e in city.edges)


def test_arterials_get_higher_speed_limit():
    city = build_city_grid(6, 6, block=50.0, seed=4, arterial_every=2, arterial_speed=25.0)
    speeds = {e.speed_limit for e in city.edges}
    assert 25.0 in speeds
    assert DEFAULT_SPEED_LIMIT in speeds
    assert any(e.speed_limit == 25.0 for e in city.edges)


def test_city_grid_is_deterministic():
    a = build_city_grid(5, 5, seed=7, jitter=0.2, one_way_prob=0.3, arterial_every=2)
    b = build_city_grid(5, 5, seed=7, jitter=0.2, one_way_prob=0.3, arterial_every=2)
    assert len(a.edges) == len(b.edges)
    assert [(e.u, e.v, round(e.length, 6)) for e in a.edges] == \
           [(e.u, e.v, round(e.length, 6)) for e in b.edges]
