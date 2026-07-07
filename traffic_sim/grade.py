"""Grade separation: an elevated highway network laid over a ground grid.

The insight that makes this cheap: the simulator's world is just a directed
graph, which need not be planar. An **overpass** is simply two edges that cross
in ``(x, y)`` but share no node — so there is no junction there (no signal, no
conflict, cars don't interact), which is exactly what an overpass is. Grade
separation therefore needs no change to the simulation core: we only add
**level-1** nodes/edges (:class:`~traffic_sim.network.Node.level`) overlaying the
ground, connected to it *only at interchanges*.

:func:`add_grade_separated` overlays an elevated **ring** (a fast loop around the
perimeter) and one elevated **expressway** (a chord across the middle row that
overpasses the ground streets it crosses), with interchanges every
``access_spacing`` metres.

**Interchanges are proper limited-access ramps**, not point junctions — the whole
point of this module's second pass. Each interchange builds, *per mainline
direction*, a separate **on-ramp** (a climb + an **acceleration lane** parallel to
the mainline, so the entering car speeds up to near highway speed before it
merges) and a separate **off-ramp** (a **deceleration lane** the exiting car peels
onto at full speed, then sheds speed *off* the through lane, then a descent).
Both taper onto/off the mainline at a **shallow angle** (< 45°) so
:func:`~traffic_sim.signals.turn_type` reads them as *straight* — which means the
per-car turn slowdown never fires on the mainline (through traffic is never
slowed by an interchange, and the exiting car does not brake in the through
lane). The deceleration lanes are sized to shed the speed drop at a comfortable
rate. The merge itself is right-of-way, reusing the roundabout mechanism: the
mainline edges are handed to :class:`~traffic_sim.priority.PriorityModel` as its
``circulating`` (priority) set, so on-ramp traffic yields to the mainline and the
mainline never yields — a car already on the highway is never stopped.

Elevated nodes are **not** added to ``node_id`` (which stays a ground
``(i, j) -> id`` map) and carry no land-use zone, so nobody parks on the highway.
Routing (Dijkstra over the whole graph) then uses the fast elevated roads for
long trips automatically. :func:`add_grade_separated` returns
``(elevated_node_ids, mainline_edge_ids)`` — the latter is the priority set.
"""

import math
from typing import Dict, List, Set, Tuple

from .network import Node, Edge, RoadNetwork, DEFAULT_SPEED_LIMIT, _ring_interchanges


def _link(net: RoadNetwork, u: int, v: int, speed: float, lanes: int,
          length=None) -> int:
    """Append one directed edge ``u -> v`` (explicit ``length`` overrides the
    geometric distance, e.g. for a ramp climb). Returns the new edge id."""
    n1, n2 = net.nodes[u], net.nodes[v]
    L = length if length is not None else math.hypot(n2.x - n1.x, n2.y - n1.y)
    eid = len(net.edges)
    net.edges.append(Edge(id=eid, u=u, v=v, length=L, lanes=lanes, speed_limit=speed))
    net.nodes[u].out_edges.append(eid)
    net.nodes[v].in_edges.append(eid)
    return eid


def _two_way(net: RoadNetwork, u: int, v: int, speed: float, lanes: int,
             length=None) -> Tuple[int, int]:
    """Append both directed edges ``u <-> v``; return the two edge ids."""
    return (_link(net, u, v, speed, lanes, length),
            _link(net, v, u, speed, lanes, length))


def add_grade_separated(net: RoadNetwork, *, block: float, speed: float,
                        lanes: int = 3, ramp_lanes: int = 2,
                        ramp_speed: float = DEFAULT_SPEED_LIMIT,
                        access_spacing: float = 1000.0,
                        merge_len: float = 70.0, diverge_len: float = 70.0,
                        taper_offset: float = 12.0, corner_radius: float = 55.0,
                        corner_fillers: int = 2,
                        ring: bool = True, expressway: bool = True
                        ) -> Tuple[Set[int], Set[int]]:
    """Overlay an elevated ring and/or expressway onto ``net`` (built ground-only).

    Adds level-1 nodes/edges at ``speed`` with ``lanes`` lanes and, at about
    every ``access_spacing`` metres, a proper interchange (see module docs):
    on-ramps with a ``merge_len`` acceleration lane and off-ramps with a
    ``diverge_len`` deceleration lane, each tapered laterally by ``taper_offset``
    so the merge/diverge is geometrically straight. Returns
    ``(elevated_node_ids, mainline_edge_ids)`` — the mainline set is the highway's
    priority (``circulating``) set for :class:`PriorityModel`.
    """
    ground = [n for n in net.nodes if n.level == 0]
    width = max(n.i for n in ground) + 1
    height = max(n.j for n in ground) + 1
    elev: Dict[Tuple[int, int], int] = {}
    mainline: Set[int] = set()

    def enode(i: int, j: int) -> int:
        """The elevated mainline node above ground ``(i, j)`` (same x, y)."""
        if (i, j) not in elev:
            g = net.nodes[net.node_id[(i, j)]]
            nid = len(net.nodes)
            net.nodes.append(Node(id=nid, i=i, j=j, x=g.x, y=g.y, level=1))
            elev[(i, j)] = nid
        return elev[(i, j)]

    def helper_node(i: int, j: int, x: float, y: float) -> int:
        """An internal level-1 ramp node at ``(x, y)`` (never zoned / parked on)."""
        nid = len(net.nodes)
        net.nodes.append(Node(id=nid, i=i, j=j, x=x, y=y, level=1, internal=True))
        return nid

    def mainline_span(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        """A two-way mainline segment between elevated nodes at ``a`` and ``b``."""
        e1, e2 = _two_way(net, enode(*a), enode(*b), speed, lanes)
        mainline.add(e1)
        mainline.add(e2)

    def interchange(i: int, j: int, fwd: Tuple[float, float],
                    lat: Tuple[float, float]) -> None:
        """Build both directions' on/off ramps at mainline node ``(i, j)``.

        ``fwd`` is a unit vector along the mainline axis, ``lat`` a unit vector
        toward the side the ramps sit on. For each travel direction ``±fwd`` an
        on-ramp (ground -> climb -> acceleration lane -> merge) and an off-ramp
        (diverge -> deceleration lane -> descent -> ground) are added; the accel/
        decel lanes taper by ``taper_offset`` so the merge/diverge is straight.
        """
        g = net.node_id[(i, j)]
        m = enode(i, j)
        mx, my = net.nodes[m].x, net.nodes[m].y
        decel_speed = 0.5 * (speed + ramp_speed)
        # Entrance lanes sit on an inner lateral band, exit lanes on an outer one,
        # so the on-ramp for one direction never lands on the off-ramp for the
        # other (they would coincide at equal lengths).
        aox, aoy = lat[0] * taper_offset, lat[1] * taper_offset
        dox, doy = lat[0] * taper_offset * 2.0, lat[1] * taper_offset * 2.0
        for sgn in (1, -1):
            fx, fy = sgn * fwd[0], sgn * fwd[1]
            # On-ramp: ground -> climb -> Acc -> accel lane -> merge at M.
            acc = helper_node(i, j, mx - fx * merge_len + aox, my - fy * merge_len + aoy)
            _link(net, g, acc, ramp_speed, ramp_lanes)          # climbing ramp
            _link(net, acc, m, speed, lanes)                    # acceleration lane
            # Off-ramp: diverge at M -> decel lane -> Dec -> descent -> ground.
            dec = helper_node(i, j, mx + fx * diverge_len + dox, my + fy * diverge_len + doy)
            _link(net, m, dec, decel_speed, lanes)              # deceleration lane
            _link(net, dec, g, ramp_speed, ramp_lanes)          # descending ramp

    spacing = max(1.0, access_spacing / block)
    ramp_rows = _ring_interchanges(height, spacing)   # left/right interchanges
    ramp_cols = _ring_interchanges(width, spacing)     # top/bottom/expressway

    def unit(ax, ay, bx, by):
        """Unit vector from ``a`` to ``b`` (zero-safe)."""
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy) or 1.0
        return dx / d, dy / d

    def corner_arc(px, py, din, dout):
        """Fillers rounding a corner at ``(px, py)`` turning from ``din`` to
        ``dout``: a short circular arc tangent to both, split into <45° chords
        (so the mainline reads *straight* through it and holds highway speed
        instead of braking to the turn speed). Returns arc points A..B."""
        r = corner_radius
        ax, ay = px - r * din[0], py - r * din[1]           # tangent point in
        bx, by = px + r * dout[0], py + r * dout[1]         # tangent point out
        s = din[0] * dout[1] - din[1] * dout[0]             # turn sign (CCW>0)
        nx, ny = (-din[1], din[0]) if s > 0 else (din[1], -din[0])
        ox, oy = ax + r * nx, ay + r * ny                   # arc centre
        a0 = math.atan2(ay - oy, ax - ox)
        a1 = math.atan2(by - oy, bx - ox)
        sweep = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        pts = [(ax, ay)]
        for f in range(1, corner_fillers + 1):
            a = a0 + sweep * f / (corner_fillers + 1)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
        pts.append((bx, by))
        return pts

    if ring:
        perim = ([(i, 0) for i in range(width)]
                 + [(width - 1, j) for j in range(1, height)]
                 + [(i, height - 1) for i in range(width - 2, -1, -1)]
                 + [(0, j) for j in range(height - 2, 0, -1)])
        # Walk the perimeter into a loop of elevated node ids, rounding each 90°
        # corner into an arc of internal filler nodes.
        loop: List[int] = []
        for k, (i, j) in enumerate(perim):
            pi, pj = perim[k - 1]
            ni, nj = perim[(k + 1) % len(perim)]
            px, py = net.nodes[net.node_id[(i, j)]].x, net.nodes[net.node_id[(i, j)]].y
            ppx, ppy = net.nodes[net.node_id[(pi, pj)]].x, net.nodes[net.node_id[(pi, pj)]].y
            nnx, nny = net.nodes[net.node_id[(ni, nj)]].x, net.nodes[net.node_id[(ni, nj)]].y
            din = unit(ppx, ppy, px, py)
            dout = unit(px, py, nnx, nny)
            if din[0] * dout[0] + din[1] * dout[1] < 0.5:    # a corner (>60° turn)
                loop.extend(helper_node(i, j, ax, ay)
                            for ax, ay in corner_arc(px, py, din, dout))
            else:
                loop.append(enode(i, j))                     # straight run / interchange
        for k in range(len(loop)):
            e1, e2 = _two_way(net, loop[k], loop[(k + 1) % len(loop)], speed, lanes)
            mainline.add(e1)
            mainline.add(e2)
        for j in ramp_rows:
            interchange(0, j, (0.0, 1.0), (1.0, 0.0))            # left side
            interchange(width - 1, j, (0.0, 1.0), (-1.0, 0.0))   # right side
        for i in ramp_cols:
            interchange(i, 0, (1.0, 0.0), (0.0, 1.0))            # bottom side
            interchange(i, height - 1, (1.0, 0.0), (0.0, -1.0))  # top side

    if expressway:
        jm = height // 2
        for i in range(width - 1):                     # a chord across the middle row
            mainline_span((i, jm), (i + 1, jm))
        for i in ramp_cols:                            # interchanges down to ground
            if 0 < i < width - 1:
                interchange(i, jm, (1.0, 0.0), (0.0, 1.0))

    elevated_nodes = {node.id for node in net.nodes if node.level == 1}
    return elevated_nodes, mainline
