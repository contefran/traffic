"""Tests for grade separation (elevated highway overlay)."""

from traffic_sim import (build_city_grid, kmh_to_ms, Car, ShortestPathRouter,
                         TrafficSim, PriorityModel)
from traffic_sim.grade import add_grade_separated
from traffic_sim.network import _strongly_connected_components
from traffic_sim.signals import turn_type, TurnType


def _build(W=10, H=10, **kw):
    net = build_city_grid(W, H, 150.0, seed=1, arterial_every=3, ring=False)
    elev, mainline = add_grade_separated(net, block=150.0, speed=kmh_to_ms(100), **kw)
    return net, elev


def _build_ml(W=12, H=12, **kw):
    net = build_city_grid(W, H, 150.0, seed=1, arterial_every=3, ring=False)
    elev, mainline = add_grade_separated(net, block=150.0, speed=kmh_to_ms(100), **kw)
    return net, elev, mainline


def test_adds_elevated_nodes_and_stays_connected():
    net, elev = _build()
    assert elev, "should have added elevated nodes"
    assert all(net.nodes[nid].level == 1 for nid in elev)
    assert all(net.nodes[n].level == 0 for n in
               [net.node_id[(i, j)] for i in range(3) for j in range(3)])
    comp = _strongly_connected_components(net.nodes, net.edges)
    assert max(comp) == 0, "grade-separated network must stay strongly connected"


def test_elevated_nodes_not_in_node_id():
    net, elev = _build()
    # node_id stays a ground-only (i, j) -> id map.
    assert all(net.nodes[nid].level == 0 for nid in net.node_id.values())
    assert not (set(net.node_id.values()) & elev)


def test_expressway_overpasses_ground_without_connecting():
    # An elevated expressway node and the ground node at the same (i, j) share
    # x, y but have no edge between them (unless it is a ramp point).
    net, elev = _build(W=10, H=10)
    jm = 10 // 2
    crossings = 0
    for nid in elev:
        n = net.nodes[nid]
        if n.j != jm or n.i in (0, 9):
            continue
        g = net.node_id[(n.i, n.j)]
        same_xy = (net.nodes[g].x, net.nodes[g].y) == (n.x, n.y)
        connected = any(net.edges[e].v == nid for e in net.nodes[g].out_edges)
        if same_xy and not connected:
            crossings += 1
    assert crossings > 0, "expressway should overpass some ground streets"


def test_ring_only_and_expressway_only():
    net_r, elev_r = _build(ring=True, expressway=False)
    net_e, elev_e = _build(ring=False, expressway=True)
    assert elev_r and elev_e
    assert max(_strongly_connected_components(net_r.nodes, net_r.edges)) == 0
    assert max(_strongly_connected_components(net_e.nodes, net_e.edges)) == 0


def test_interchange_merges_and_diverges_are_straight():
    # A tapered on/off ramp must read as a *straight* movement (turn_type < 45°)
    # so the per-car turn slowdown never fires on the mainline. Check every
    # accel-lane -> mainline merge and mainline -> decel-lane diverge along the
    # direction the router would actually commit to (the aligned one).
    import math
    net, elev, mainline = _build_ml()

    def head(eid):
        e = net.edges[eid]
        u, v = net.nodes[e.u], net.nodes[e.v]
        return math.atan2(v.y - u.y, v.x - u.x)

    def aligned(cands, h):
        return min(cands, key=lambda e: abs((head(e) - h + math.pi) % (2 * math.pi) - math.pi))

    merges = diverges = 0
    for e in net.edges:
        u, v = net.nodes[e.u], net.nodes[e.v]
        if u.internal and u.level == 1 and v.level == 1 and not v.internal:   # accel -> M
            outs = [o for o in net.nodes[e.v].out_edges if o in mainline]
            if outs:
                assert turn_type(net, e.id, aligned(outs, head(e.id))) is TurnType.STRAIGHT
                merges += 1
        if not u.internal and u.level == 1 and v.internal and v.level == 1:   # M -> decel
            ins = [i for i in net.nodes[e.u].in_edges if i in mainline]
            if ins:
                assert turn_type(net, aligned(ins, head(e.id)), e.id) is TurnType.STRAIGHT
                diverges += 1
    assert merges > 0 and diverges > 0


def test_mainline_traffic_has_priority_and_a_lone_car_holds_speed():
    # On-ramp traffic yields to the mainline (mainline = the priority set), and a
    # lone car driving the ring holds highway speed the whole way (rounded corners
    # + straight merges mean the turn slowdown never bites).
    net, elev, mainline = _build_ml()
    pri = PriorityModel(net, circulating=mainline)
    # A car already on the mainline never yields; entering traffic can.
    ml_edge = min(mainline)
    assert pri.must_yield(ml_edge, next(iter(
        e for e in net.nodes[net.edges[ml_edge].v].out_edges if e in mainline)), []) is False

    car = Car(id=0, edge_id=ml_edge, s=0.0, v=kmh_to_ms(100))
    sim = TrafficSim(net, [car], ShortestPathRouter(net, seed=0), priority=pri)
    speeds = []
    for _ in range(3000):
        sim.step(0.1)
        if car.edge_id in mainline:
            speeds.append(car.v)
    assert speeds and min(speeds) > kmh_to_ms(90), "lone car should hold highway speed"
    assert sim.crashes == 0
