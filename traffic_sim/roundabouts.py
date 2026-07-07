"""Geometric roundabouts: replace an intersection with a real circulating ring.

Unlike a signalized or priority-controlled crossing (which stays a single point),
a roundabout is built **as geometry**: a ring of one-way edges around the old
junction, with each approach street cut back to an entry/exit point on the ring.
Cars physically drive *around* the circle to their exit — the routing (Dijkstra)
discovers that for free, since it is just more edges. A U-turn falls out naturally
(go all the way around); an immediate exit back the way you came is the forbidden
reverse of the entry edge, so a car must circulate.

Three things make it behave like a roundabout, all reusing existing machinery:

* **Slow** — ring edges carry a low ``ring_speed`` and every movement through the
  circle is a turn, so the turn-approach slowdown already brakes cars to it.
* **Entry yields to circulating traffic** — the ring nodes are unsignalized and
  the :class:`~traffic_sim.priority.PriorityModel` is told which edges are
  *circulating*; a car entering the ring gives way to a car already on it.
* **Not a place** — ring nodes (and the disconnected centre island) are marked
  ``Node.internal``, so they are never zoned, never a destination, never parked
  on.

``add_roundabout(net, node_id, ...)`` converts one intersection in place and
returns ``(ring_node_ids, circulating_edge_ids)``; ``add_roundabouts`` does a
batch and unions the results. Apply **after** the grid is built (like grade
separation) — it preserves reachability (every approach still reaches every other
via the one-way ring).
"""

import math
from typing import Dict, List, Set, Tuple

from .network import Node, Edge, RoadNetwork, DEFAULT_SPEED_LIMIT

# Default circulating speed inside a roundabout (~25 km/h in m/s).
DEFAULT_RING_SPEED = 25.0 / 3.6


def _add_node(net: RoadNetwork, x: float, y: float, i: int, j: int) -> int:
    """Append an internal ground node at ``(x, y)`` and return its id."""
    nid = len(net.nodes)
    net.nodes.append(Node(id=nid, i=i, j=j, x=x, y=y, level=0, internal=True))
    return nid


def _add_edge(net: RoadNetwork, u: int, v: int, speed: float, lanes: int) -> int:
    """Append a one-way edge ``u -> v`` (length from geometry) and return its id."""
    n1, n2 = net.nodes[u], net.nodes[v]
    eid = len(net.edges)
    net.edges.append(Edge(id=eid, u=u, v=v,
                          length=math.hypot(n2.x - n1.x, n2.y - n1.y),
                          lanes=lanes, speed_limit=speed))
    net.nodes[u].out_edges.append(eid)
    net.nodes[v].in_edges.append(eid)
    return eid


def _retarget(net: RoadNetwork, eid: int, new_v: int) -> None:
    """Redirect edge ``eid`` (…->old_v) to end at ``new_v`` instead."""
    e = net.edges[eid]
    net.nodes[e.v].in_edges.remove(eid)
    e.v = new_v
    net.nodes[new_v].in_edges.append(eid)
    e.length = math.hypot(net.nodes[e.v].x - net.nodes[e.u].x,
                          net.nodes[e.v].y - net.nodes[e.u].y)


def _resource(net: RoadNetwork, eid: int, new_u: int) -> None:
    """Redirect edge ``eid`` (old_u->…) to start at ``new_u`` instead."""
    e = net.edges[eid]
    net.nodes[e.u].out_edges.remove(eid)
    e.u = new_u
    net.nodes[new_u].out_edges.append(eid)
    e.length = math.hypot(net.nodes[e.v].x - net.nodes[e.u].x,
                          net.nodes[e.v].y - net.nodes[e.u].y)


def add_roundabout(net: RoadNetwork, center_id: int, *, radius: float = 18.0,
                   ring_speed: float = DEFAULT_RING_SPEED, ring_lanes: int = 1,
                   fillers: int = 1) -> Tuple[Set[int], Set[int]]:
    """Turn intersection ``center_id`` into a geometric roundabout, in place.

    A ring of ``fillers``-subdivided one-way edges (counter-clockwise, right-hand
    traffic) is built at ``radius`` metres around the junction; each approach is
    cut back to its own point on the ring (entry *and* exit share that point).
    ``ring_speed``/``ring_lanes`` govern the circulating road; ``fillers`` extra
    nodes per arc round out the circle. The centre node is disconnected and marked
    internal. Returns ``(ring_node_ids, circulating_edge_ids)``.
    """
    centre = net.nodes[center_id]
    xc, yc, ci, cj = centre.x, centre.y, centre.i, centre.j

    # Gather approaches: neighbour node -> {'in': A->X edge, 'out': X->A edge}.
    approaches: Dict[int, Dict[str, int]] = {}
    for eid in list(centre.in_edges):
        approaches.setdefault(net.edges[eid].u, {})["in"] = eid
    for eid in list(centre.out_edges):
        approaches.setdefault(net.edges[eid].v, {})["out"] = eid
    if len(approaches) < 2:
        return set(), set()   # nothing to circulate

    # Order approaches by bearing from the centre.
    items = sorted(
        (math.atan2(net.nodes[nb].y - yc, net.nodes[nb].x - xc), nb, io)
        for nb, io in approaches.items())

    # Lay ring nodes counter-clockwise: one at each approach bearing, plus
    # ``fillers`` evenly between consecutive approaches for a round shape.
    ring: List[Tuple[int, int]] = []   # (node id, approach index or -1)
    n = len(items)
    for k, (theta, _nb, _io) in enumerate(items):
        rid = _add_node(net, xc + radius * math.cos(theta),
                        yc + radius * math.sin(theta), ci, cj)
        ring.append((rid, k))
        gap = (items[(k + 1) % n][0] - theta) % (2.0 * math.pi)
        for f in range(1, fillers + 1):
            a = theta + gap * f / (fillers + 1)
            ring.append((_add_node(net, xc + radius * math.cos(a),
                                   yc + radius * math.sin(a), ci, cj), -1))

    ring_nodes = {rid for rid, _ in ring}

    # One-way circulating edges around the ring (they are consecutive in CCW order).
    circulating: Set[int] = set()
    for k in range(len(ring)):
        u = ring[k][0]
        v = ring[(k + 1) % len(ring)][0]
        circulating.add(_add_edge(net, u, v, ring_speed, ring_lanes))

    # Re-attach each approach to its own ring node (entry in, exit out).
    approach_node = {k: rid for rid, k in ring if k >= 0}
    for k, (_theta, _nb, io) in enumerate(items):
        rid = approach_node[k]
        if "in" in io:
            _retarget(net, io["in"], rid)    # A->X  becomes  A->ring
        if "out" in io:
            _resource(net, io["out"], rid)    # X->A  becomes  ring->A

    # The old centre is now an isolated island — no traffic, not a place.
    centre.internal = True
    centre.in_edges.clear()
    centre.out_edges.clear()
    return ring_nodes, circulating


def add_roundabouts(net: RoadNetwork, center_ids, **kwargs) -> Tuple[Set[int], Set[int]]:
    """Convert several intersections into roundabouts; union the returned sets."""
    all_nodes: Set[int] = set()
    all_edges: Set[int] = set()
    for cid in center_ids:
        nodes, edges = add_roundabout(net, cid, **kwargs)
        all_nodes |= nodes
        all_edges |= edges
    return all_nodes, all_edges
