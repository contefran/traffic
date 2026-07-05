"""Grade separation: an elevated highway network laid over a ground grid.

The insight that makes this cheap: the simulator's world is just a directed
graph, which need not be planar. An **overpass** is simply two edges that cross
in ``(x, y)`` but share no node — so there is no junction there (no signal, no
conflict, cars don't interact), which is exactly what an overpass is. Grade
separation therefore needs no change to the simulation core: we only add
**level-1** nodes/edges (:class:`~traffic_sim.network.Node.level`) overlaying the
ground, connected to it *only at ramps*.

:func:`add_grade_separated` overlays an elevated **ring** (a fast loop around the
perimeter) and one elevated **expressway** (a chord across the middle row that
overpasses the ground streets it crosses), with ramps every ``access_spacing``
metres. Elevated nodes are **not** added to ``node_id`` (which stays a ground
``(i, j) -> id`` map) and carry no land-use zone, so nobody parks on the highway.
Routing (Dijkstra over the whole graph) then uses the fast elevated roads for
long trips automatically.
"""

import math
from typing import Dict, Set, Tuple

from .network import Node, Edge, RoadNetwork, DEFAULT_SPEED_LIMIT, _ring_interchanges


def _link(net: RoadNetwork, u: int, v: int, speed: float, lanes: int,
          length=None) -> None:
    """Append one directed edge ``u -> v`` (explicit ``length`` for ramps)."""
    n1, n2 = net.nodes[u], net.nodes[v]
    L = length if length is not None else math.hypot(n2.x - n1.x, n2.y - n1.y)
    eid = len(net.edges)
    net.edges.append(Edge(id=eid, u=u, v=v, length=L, lanes=lanes, speed_limit=speed))
    net.nodes[u].out_edges.append(eid)
    net.nodes[v].in_edges.append(eid)


def _two_way(net: RoadNetwork, u: int, v: int, speed: float, lanes: int,
             length=None) -> None:
    _link(net, u, v, speed, lanes, length)
    _link(net, v, u, speed, lanes, length)


def add_grade_separated(net: RoadNetwork, *, block: float, speed: float,
                        lanes: int = 3, ramp_lanes: int = 2, ramp_length: float = 45.0,
                        ramp_speed: float = DEFAULT_SPEED_LIMIT,
                        access_spacing: float = 1000.0,
                        ring: bool = True, expressway: bool = True) -> Set[int]:
    """Overlay an elevated ring and/or expressway onto ``net`` (built ground-only).

    Adds level-1 nodes/edges at ``speed`` with ``lanes`` lanes, ramps (``ramp_
    lanes``/``ramp_length``) linking them to the ground about every
    ``access_spacing`` metres. Returns the set of new elevated node ids.
    """
    ground = [n for n in net.nodes if n.level == 0]
    width = max(n.i for n in ground) + 1
    height = max(n.j for n in ground) + 1
    elev: Dict[Tuple[int, int], int] = {}

    def enode(i: int, j: int) -> int:
        """The elevated node above ground ``(i, j)`` (created once, same x, y)."""
        if (i, j) not in elev:
            g = net.nodes[net.node_id[(i, j)]]
            nid = len(net.nodes)
            net.nodes.append(Node(id=nid, i=i, j=j, x=g.x, y=g.y, level=1))
            elev[(i, j)] = nid
        return elev[(i, j)]

    def ramp(i: int, j: int) -> None:
        """A two-way ramp between ground and elevated at ``(i, j)``. Slower than
        the highway, so the priority model gives merging highway traffic
        right of way over cars coming up the ramp."""
        _two_way(net, net.node_id[(i, j)], enode(i, j), ramp_speed, ramp_lanes, ramp_length)

    spacing = max(1.0, access_spacing / block)
    ramp_rows = _ring_interchanges(height, spacing)   # left/right ramps at these rows
    ramp_cols = _ring_interchanges(width, spacing)     # top/bottom/expressway ramps

    if ring:
        perim = ([(i, 0) for i in range(width)]
                 + [(width - 1, j) for j in range(1, height)]
                 + [(i, height - 1) for i in range(width - 2, -1, -1)]
                 + [(0, j) for j in range(height - 2, 0, -1)])
        for k in range(len(perim)):
            _two_way(net, enode(*perim[k]), enode(*perim[(k + 1) % len(perim)]), speed, lanes)
        for j in ramp_rows:
            ramp(0, j)
            ramp(width - 1, j)
        for i in ramp_cols:
            ramp(i, 0)
            ramp(i, height - 1)

    if expressway:
        jm = height // 2
        for i in range(width - 1):                     # a chord across the middle row
            _two_way(net, enode(i, jm), enode(i + 1, jm), speed, lanes)
        for i in ramp_cols:                            # ramps down to the ground
            if 0 < i < width - 1:
                ramp(i, jm)

    return set(elev.values())
