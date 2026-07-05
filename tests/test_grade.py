"""Tests for grade separation (elevated highway overlay)."""

from traffic_sim import build_city_grid, kmh_to_ms
from traffic_sim.grade import add_grade_separated
from traffic_sim.network import _strongly_connected_components


def _build(W=10, H=10, **kw):
    net = build_city_grid(W, H, 150.0, seed=1, arterial_every=3, ring=False)
    elev = add_grade_separated(net, block=150.0, speed=kmh_to_ms(100), **kw)
    return net, elev


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
