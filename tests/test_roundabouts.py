"""Tests for geometric roundabouts (a real circulating ring)."""

from traffic_sim import (
    build_grid_network, add_roundabout, assign_zones, edges_by_zone, LandUse,
    Car, ShortestPathRouter, TrafficSim, PriorityModel,
)


def _out(net, a, b):
    return next(e for e in net.nodes[a].out_edges if net.edges[e].v == b)


def test_construction_islands_the_centre_and_builds_a_one_way_ring():
    net = build_grid_network(5, 5, block=150.0)
    X = net.node_id[(2, 2)]
    approaches = len(net.nodes[X].in_edges)          # 4-way -> 4 in, 4 out
    ring_nodes, circ = add_roundabout(net, X, radius=18.0, fillers=1)

    # Centre becomes an internal island with no edges.
    assert net.nodes[X].internal
    assert not net.nodes[X].in_edges and not net.nodes[X].out_edges
    # 4 approach ring nodes + 1 filler each = 8 nodes / 8 circulating edges.
    assert len(ring_nodes) == approaches * 2
    assert len(circ) == approaches * 2
    assert all(net.nodes[r].internal for r in ring_nodes)
    # Circulating edges form a single directed cycle through the ring nodes.
    for r in ring_nodes:
        outs = [e for e in net.nodes[r].out_edges if e in circ]
        ins = [e for e in net.nodes[r].in_edges if e in circ]
        assert len(outs) == 1 and len(ins) == 1


def test_a_car_routes_through_the_roundabout_and_exits():
    net = build_grid_network(5, 5, block=150.0)
    X = net.node_id[(2, 2)]
    _, circ = add_roundabout(net, X, radius=18.0, fillers=1)
    start = _out(net, net.node_id[(0, 2)], net.node_id[(1, 2)])
    far = net.node_id[(4, 2)]
    car = Car(id=0, edge_id=start, s=0.0, v=8.0, dest=far)
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0))
    used_ring = False
    for _ in range(800):
        sim.step(0.1)
        if car.edge_id in circ:
            used_ring = True
        if car.edge_id in net.nodes[far].in_edges:
            break
    assert used_ring, "the car should have driven around the ring"
    assert car.edge_id in net.nodes[far].in_edges, "and reached the far side"


def test_entry_yields_to_circulating_but_circulating_never_yields():
    net = build_grid_network(5, 5, block=150.0)
    X = net.node_id[(2, 2)]
    ring_nodes, circ = add_roundabout(net, X, radius=18.0, fillers=1)
    model = PriorityModel(net, circulating=circ)

    # Pick a ring node with both an entry (non-circulating in) and a circulating in.
    r = next(n for n in ring_nodes
             if any(e not in circ for e in net.nodes[n].in_edges))
    entry = next(e for e in net.nodes[r].in_edges if e not in circ)
    circ_in = next(e for e in net.nodes[r].in_edges if e in circ)
    ring_out = next(e for e in net.nodes[r].out_edges if e in circ)

    # An entering car (its committed move is onto the ring) yields to an imminent
    # circulating car...
    contenders = [(circ_in, ring_out, 2.0, 6.0)]     # circulating car right there
    assert model.must_yield(entry, ring_out, contenders) is True
    # ...but not when the ring is clear.
    assert model.must_yield(entry, ring_out, []) is False
    # A circulating car never yields (it has absolute priority).
    assert model.must_yield(circ_in, ring_out, [(entry, ring_out, 1.0, 6.0)]) is False


def test_ring_streets_are_never_zoned_or_destinations():
    net = build_grid_network(6, 6, block=150.0)
    X = net.node_id[(3, 3)]
    _, circ = add_roundabout(net, X, radius=18.0, fillers=1)
    zones = assign_zones(net, seed=0)
    assert circ.isdisjoint(zones), "ring edges must be unzoned"
    router = ShortestPathRouter(net, seed=0)
    assert circ.isdisjoint(router._dest_edges), "ring edges are never destinations"
