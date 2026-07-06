"""Tests for destination-based routing (ShortestPathRouter)."""

import math

from traffic_sim import (
    build_grid_network,
    build_city_grid,
    Car,
    ShortestPathRouter,
    TrafficSim,
)


def _edge_between(net, a, b):
    """The edge id from node ``a`` to node ``b`` (assumes adjacency)."""
    for eid in net.nodes[a].out_edges:
        if net.edges[eid].v == b:
            return eid
    raise AssertionError(f"no edge {a}->{b}")


def test_cost_to_go_matches_manhattan_on_uniform_grid():
    # On a uniform grid every edge has equal free-flow cost, so cost-to-go is
    # proportional to Manhattan distance in grid steps.
    net = build_grid_network(width=4, height=4, block=50.0)
    router = ShortestPathRouter(net, seed=0)
    dest = net.node_id[(3, 3)]
    dist = router._dist_to(dest)

    step_cost = 50.0 / net.edges[0].speed_limit
    for (i, j), nid in net.node_id.items():
        steps = abs(3 - i) + abs(3 - j)
        assert math.isclose(dist[nid], steps * step_cost, rel_tol=1e-9)


def test_router_steps_toward_destination():
    net = build_grid_network(width=4, height=4, block=50.0)
    router = ShortestPathRouter(net, seed=0)
    dest = net.node_id[(3, 3)]

    # Start at bottom-left heading right; destination is top-right corner.
    start = net.node_id[(0, 0)]
    nbr = net.node_id[(1, 0)]
    car = Car(id=0, edge_id=_edge_between(net, start, nbr), s=0.0, v=0.0, dest=dest)

    dist_before = router._dist_to(dest)[nbr]
    nxt = router.next_edge(car.edge_id, car)
    # The chosen edge must strictly reduce the remaining cost to the destination.
    arrive = net.edges[nxt].v
    assert router._dist_to(dest)[arrive] < dist_before


def test_car_actually_reaches_its_destination():
    net = build_grid_network(width=4, height=4, block=50.0)
    router = ShortestPathRouter(net, seed=0)
    dest = net.node_id[(3, 3)]

    start = net.node_id[(0, 0)]
    nbr = net.node_id[(1, 0)]
    car = Car(id=0, edge_id=_edge_between(net, start, nbr), s=0.0, v=0.0, dest=dest)
    sim = TrafficSim(net, [car], router)

    reached = False
    for _ in range(2000):
        sim.step(0.1)
        # The car arrives when it is sitting at the start of an edge leaving dest
        # (the router reassigns on arrival, so we watch the edge it departs on).
        if net.edges[car.edge_id].u == dest and car.s < 1.0:
            reached = True
            break
    assert reached


def test_router_prefers_faster_arterial_route():
    # A grid where row 0 is a fast arterial. Going (0,0) -> (3,0) along the
    # arterial should be cheaper than detouring through slow interior streets,
    # so the first hop from (0,0) toward (3,0) stays on the arterial row.
    net = build_city_grid(width=4, height=4, block=50.0, seed=0,
                          arterial_every=4, arterial_speed=30.0)  # only index 0
    router = ShortestPathRouter(net, seed=0)
    dest = net.node_id[(3, 0)]

    start = net.node_id[(0, 0)]
    # Approach (0,0) from above so both "continue along arterial" and "leave it"
    # are live options at the node.
    above = net.node_id[(0, 1)]
    car = Car(id=0, edge_id=_edge_between(net, above, start), s=0.0, v=0.0, dest=dest)

    nxt = router.next_edge(car.edge_id, car)
    # Fastest route hugs the arterial (row j == 0).
    assert net.nodes[net.edges[nxt].v].j == 0


def test_dead_end_returns_none():
    net = build_grid_network(width=1, height=2, block=50.0)
    router = ShortestPathRouter(net, seed=0)
    # Edge into the top node, which only connects back down (its lone out-edge
    # is the U-turn). next_edge should still return that single option, not crash.
    top = net.node_id[(0, 1)]
    bottom = net.node_id[(0, 0)]
    car = Car(id=0, edge_id=_edge_between(net, bottom, top), s=0.0, v=0.0, dest=bottom)
    nxt = router.next_edge(car.edge_id, car)
    assert nxt in net.nodes[top].out_edges


def test_router_never_makes_u_turn_even_toward_destination():
    # The destination sits directly behind the car (the node it just left), so a
    # U-turn would be the single cheapest move. U-turns are forbidden, so the
    # router must route around rather than reverse straight back.
    net = build_grid_network(width=3, height=3, block=50.0)
    router = ShortestPathRouter(net, seed=0)

    behind = net.node_id[(1, 0)]
    here = net.node_id[(1, 1)]
    car = Car(id=0, edge_id=_edge_between(net, behind, here), s=0.0, v=0.0, dest=behind)

    nxt = router.next_edge(car.edge_id, car)
    assert net.edges[nxt].v != behind          # not the reverse (U-turn) edge
    assert nxt in net.nodes[here].out_edges     # a genuine forward option


def test_assigns_destination_when_missing():
    net = build_grid_network(width=3, height=3, block=50.0)
    router = ShortestPathRouter(net, seed=1)
    start = net.node_id[(0, 0)]
    nbr = net.node_id[(1, 0)]
    car = Car(id=0, edge_id=_edge_between(net, start, nbr), s=0.0, v=0.0)  # no dest
    router.next_edge(car.edge_id, car)
    # A fresh destination street was assigned (never the car's current edge), and
    # the routing target is that street's upstream node.
    assert car.dest_edge is not None and car.dest_edge != car.edge_id
    assert car.dest == net.edges[car.dest_edge].u


def test_deterministic_with_seed():
    net = build_city_grid(width=5, height=5, block=50.0, seed=2,
                          jitter=0.1, one_way_prob=0.1, drop_prob=0.1,
                          arterial_every=2, arterial_speed=25.0)

    def run(seed):
        router = ShortestPathRouter(net, seed=seed)
        cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(10)]
        sim = TrafficSim(net, cars, router)
        for _ in range(300):
            sim.step(0.1)
        return [(c.edge_id, round(c.s, 6), c.dest) for c in cars]

    assert run(5) == run(5)
