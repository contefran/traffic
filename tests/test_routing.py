from traffic_sim import build_grid_network, RandomRouter


def test_router_is_deterministic_with_seed():
    net = build_grid_network(width=3, height=3, block=50.0)
    a = RandomRouter(net, seed=7)
    b = RandomRouter(net, seed=7)
    seq_a = [a.next_edge(0) for _ in range(20)]
    seq_b = [b.next_edge(0) for _ in range(20)]
    assert seq_a == seq_b


def test_router_avoids_uturn_when_possible():
    net = build_grid_network(width=3, height=3, block=50.0)
    router = RandomRouter(net, seed=1, allow_uturn=False)
    # For any edge with an alternative, the chosen edge must not lead straight back.
    for edge in net.edges:
        for _ in range(10):
            nxt = router.next_edge(edge.id)
            if nxt is None:
                continue
            node = net.nodes[edge.v]
            has_alternative = any(net.edges[eid].v != edge.u for eid in node.out_edges)
            if has_alternative:
                assert net.edges[nxt].v != edge.u


def test_router_returns_valid_out_edge():
    net = build_grid_network(width=3, height=3, block=50.0)
    router = RandomRouter(net, seed=3)
    for edge in net.edges:
        nxt = router.next_edge(edge.id)
        assert nxt in net.nodes[edge.v].out_edges
