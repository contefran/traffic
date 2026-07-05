"""Tests for the multi-lane road structure (ring / arterial / local lane counts)."""

from traffic_sim import build_city_grid, build_grid_network, Car, RandomRouter, TrafficSim, kmh_to_ms
from traffic_sim.network import DEFAULT_SPEED_LIMIT


def _edge(net, a, b):
    aid, bid = net.node_id[a], net.node_id[b]
    return next((e for e in net.edges if e.u == aid and e.v == bid), None)


def test_ring_arterial_local_lane_counts():
    net = build_city_grid(8, 8, 150.0, seed=1, arterial_every=3,
                          arterial_speed=kmh_to_ms(70), arterial_lanes=2,
                          ring=True, ring_speed=kmh_to_ms(100), ring_lanes=3)
    counts = {e.lanes for e in net.edges}
    assert {1, 2, 3} <= counts

    top = _edge(net, (3, 7), (4, 7))          # top perimeter -> ring
    assert top.lanes == 3
    assert abs(top.speed_limit - kmh_to_ms(100)) < 1e-6

    # An arterial row/col (index 3) that is not on the border -> 2 lanes.
    art = _edge(net, (1, 3), (2, 3))
    assert art.lanes == 2

    interior_local = _edge(net, (2, 2), (2, 1)) or _edge(net, (2, 1), (2, 2))
    assert interior_local.lanes == 1
    assert abs(interior_local.speed_limit - DEFAULT_SPEED_LIMIT) < 1e-6


def test_no_ring_by_default():
    net = build_city_grid(6, 6, 150.0, seed=2, arterial_every=3)
    assert all(e.lanes <= 2 for e in net.edges)     # no 3-lane ring
    # Border edges are ordinary here (not a fast ring).
    top = _edge(net, (2, 5), (3, 5))
    assert top is not None and top.lanes <= 2


def test_ring_edges_are_never_dropped():
    # Even with aggressive drops the full perimeter loop survives (protected).
    net = build_city_grid(6, 6, 150.0, seed=3, drop_prob=0.6, ring=True)
    border = [((i, 0), (i + 1, 0)) for i in range(5)]        # bottom row
    for a, b in border:
        assert _edge(net, a, b) is not None or _edge(net, b, a) is not None


def test_ring_is_limited_access_but_still_connected():
    # On/off ramps are few (at least one per side) and the graph stays strongly
    # connected via the ring loop.
    from traffic_sim.network import _strongly_connected_components
    W = Hh = 8
    net = build_city_grid(W, Hh, 150.0, seed=1, jitter=0.15, one_way_prob=0.15,
                          drop_prob=0.12, arterial_every=3, ring=True,
                          ring_access_spacing=1000.0)
    is_border = lambda n: n.i in (0, W - 1) or n.j in (0, Hh - 1)
    ramps = sum(1 for e in net.edges
                if is_border(net.nodes[e.u]) and not is_border(net.nodes[e.v]))
    # Far fewer than one ramp per border node (there are dozens of border nodes).
    assert 4 <= ramps <= 12
    comp = _strongly_connected_components(net.nodes, net.edges)
    assert max(comp) == 0, "every destination must stay reachable"


def test_tighter_spacing_gives_more_ramps():
    W = Hh = 10
    def ramps(spacing):
        net = build_city_grid(W, Hh, 150.0, seed=1, ring=True,
                              ring_access_spacing=spacing)
        b = lambda n: n.i in (0, W - 1) or n.j in (0, Hh - 1)
        return sum(1 for e in net.edges
                   if b(net.nodes[e.u]) and not b(net.nodes[e.v]))
    assert ramps(400.0) > ramps(1500.0)


def test_lanes_decouple_car_following():
    # A car in lane 1 is not blocked by a stopped car in lane 0 on the same edge.
    net = build_grid_network(3, 3, block=150.0)
    net.edges[0].lanes = 2
    blocker = Car(id=0, edge_id=0, s=70.0, v=0.0, lane=0)
    mover = Car(id=1, edge_id=0, s=40.0, v=10.0, lane=1)
    sim = TrafficSim(net, [blocker, mover], RandomRouter(net, seed=0))
    for _ in range(30):
        sim.step(0.1)
    # The lane-1 car sailed past the lane-0 blocker's position (or off the edge).
    assert mover.edge_id != 0 or mover.s > 70.0
    assert sim.crashes == 0


def test_car_following_holds_when_all_lanes_blocked():
    # With both lanes blocked at the same point there is no escape, so the
    # follower must queue behind (car-following still holds within a lane).
    net = build_grid_network(3, 3, block=150.0)
    net.edges[0].lanes = 2
    b0 = Car(id=0, edge_id=0, s=70.0, v=0.0, lane=0)
    b1 = Car(id=1, edge_id=0, s=70.0, v=0.0, lane=1)
    follower = Car(id=2, edge_id=0, s=40.0, v=10.0, lane=1)
    sim = TrafficSim(net, [b0, b1, follower], RandomRouter(net, seed=0))
    for _ in range(30):
        sim.step(0.1)
    if follower.edge_id == 0:
        assert follower.s <= 70.0 - follower.length          # cannot pass
    assert sim.crashes == 0


def test_fast_car_changes_lane_to_overtake():
    # A fast car stuck behind a slow one moves to the open lane, passes, and
    # (keep-right) drifts back. It ends up ahead of the slow car.
    net = build_grid_network(3, 3, block=300.0)
    net.edges[0].lanes = 2
    slow = Car(id=0, edge_id=0, s=120.0, v=3.0, lane=0)
    slow.max_speed = 3.0                                     # genuinely slow
    fast = Car(id=1, edge_id=0, s=80.0, v=13.0, lane=0)      # behind, same lane
    sim = TrafficSim(net, [slow, fast], RandomRouter(net, seed=0))
    changed_lane = False
    for _ in range(60):
        sim.step(0.1)
        if fast.edge_id == 0 and fast.lane != 0:
            changed_lane = True
    assert changed_lane, "fast car should have used the other lane"
    assert fast.edge_id != 0 or fast.s > slow.s              # overtook
    assert sim.crashes == 0


def test_transfer_clamps_lane_to_next_edge():
    # A car in lane 2 moving onto a 1-lane edge ends up in lane 0 (clamped).
    net = build_grid_network(3, 3, block=150.0)
    start = net.node_id[(1, 1)]
    nbr = net.node_id[(2, 1)]
    approach = _edge_id(net, net.node_id[(0, 1)], start)
    net.edges[approach].lanes = 3
    car = Car(id=0, edge_id=approach, s=149.0, v=12.0, lane=2)
    car.next_edge = _edge_id(net, start, nbr)                 # a 1-lane local edge
    sim = TrafficSim(net, [car], RandomRouter(net, seed=0))
    sim.step(0.5)
    assert car.edge_id == _edge_id(net, start, nbr)           # transferred
    assert car.lane == 0                                      # clamped into lane 0


def _edge_id(net, a, b):
    return next(eid for eid in net.nodes[a].out_edges if net.edges[eid].v == b)
