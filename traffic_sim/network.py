"""Static road-network model: nodes, edges, and a grid builder.

The network is a directed graph. A neighbouring pair of grid points is
connected by *two* directed edges (one per direction), so traffic can flow
both ways. A vehicle's world position is always derived from an edge plus a
distance ``s`` along it via :meth:`RoadNetwork.point_on_edge` — geometry lives
here and nowhere else.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import random

# Default speed limit: ~50 km/h expressed in m/s.
DEFAULT_SPEED_LIMIT = 13.9

# Lane width [m], used only to lay lanes out side by side for rendering.
LANE_WIDTH = 3.5

# Rendering-only offset for elevated (level-1) nodes, so overpasses draw beside
# the ground road they cross rather than on top of it.
LEVEL_OFFSET = 16.0


@dataclass
class Node:
    """A grid intersection (graph vertex).

    Carries both its integer grid indices ``(i, j)`` — which the H/V signal
    model depends on and which stay fixed even when positions are jittered — and
    its world position ``(x, y)`` in metres. ``level`` is the grade: 0 = ground,
    1 = elevated (an overpass). Two nodes can share an ``(x, y)`` at different
    levels; because they are distinct nodes with no edge between them, the roads
    *cross without connecting* — the whole basis of grade separation.
    ``out_edges`` / ``in_edges`` hold the ids of edges leaving / entering this
    node (indices into :attr:`RoadNetwork.edges`).
    """

    id: int
    i: int  # grid column
    j: int  # grid row
    x: float
    y: float
    level: int = 0  # 0 = ground, 1 = elevated
    # An *internal* node is part of a junction's machinery (a roundabout ring
    # node, or the disconnected island at its centre) rather than a place: it is
    # never zoned, never a trip destination, and nobody parks there.
    internal: bool = False
    out_edges: List[int] = field(default_factory=list)
    in_edges: List[int] = field(default_factory=list)


@dataclass
class Edge:
    """A one-way road segment from node ``u`` to node ``v``.

    ``length`` is in metres and ``speed_limit`` in m/s; a two-way street is
    represented by two opposite ``Edge`` objects. ``lanes`` is carried for a
    future multi-lane model but is not yet used by the dynamics.
    """

    id: int
    u: int  # source node id
    v: int  # target node id
    length: float
    lanes: int = 1
    speed_limit: float = DEFAULT_SPEED_LIMIT  # [m/s]


@dataclass
class RoadNetwork:
    """A directed road graph plus the geometry needed to place cars on it.

    ``nodes`` and ``edges`` are indexed by id (their list position); ``node_id``
    maps a grid coordinate ``(i, j)`` to its node id. This object is the single
    source of world geometry — a car's ``(x, y)`` always comes from
    :meth:`point_on_edge`, never stored on the car.
    """

    nodes: List[Node]
    edges: List[Edge]
    node_id: Dict[Tuple[int, int], int]  # (i, j) -> node id

    def point_on_edge(self, edge_id: int, s: float) -> Tuple[float, float]:
        """World (x, y) of a point ``s`` metres along ``edge_id`` (centreline)."""
        e = self.edges[edge_id]
        n1, n2 = self.nodes[e.u], self.nodes[e.v]
        t = s / e.length if e.length > 0 else 0.0
        return (n1.x + t * (n2.x - n1.x), n1.y + t * (n2.y - n1.y))

    def point_on_edge_lane(self, edge_id: int, s: float, lane: int) -> Tuple[float, float]:
        """World (x, y) of a car in ``lane`` at ``s`` along ``edge_id``.

        Offsets the centreline point sideways so each lane draws in its own
        track: lanes are numbered 0 (rightmost) upward, spaced ``LANE_WIDTH``
        apart, laid out to the right of the direction of travel. Rendering only —
        the dynamics work in ``(edge, lane, s)``.
        """
        e = self.edges[edge_id]
        cx, cy = self.point_on_edge(edge_id, s)
        if e.length <= 0:
            return cx, cy
        n1, n2 = self.nodes[e.u], self.nodes[e.v]
        dx, dy = (n2.x - n1.x) / e.length, (n2.y - n1.y) / e.length
        # Right-hand perpendicular to the heading; lanes stack from the centre.
        px, py = dy, -dx
        offset = (lane - (e.lanes - 1) / 2.0) * LANE_WIDTH
        # Elevated level offset, interpolated so ramps slope up smoothly.
        t = s / e.length
        lvl = n1.level * (1.0 - t) + n2.level * t
        lift = lvl * LEVEL_OFFSET
        return cx + px * offset + lift, cy + py * offset + lift

    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) over all nodes."""
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return min(xs), min(ys), max(xs), max(ys)


def build_grid_network(width: int, height: int, block: float) -> RoadNetwork:
    """Build a ``width`` x ``height`` grid with two-way edges between neighbours.

    ``block`` is the spacing between adjacent grid points, in metres.
    """
    nodes: List[Node] = []
    node_id: Dict[Tuple[int, int], int] = {}

    nid = 0
    for j in range(height):
        for i in range(width):
            node_id[(i, j)] = nid
            nodes.append(Node(id=nid, i=i, j=j, x=i * block, y=j * block))
            nid += 1

    edges: List[Edge] = []

    def add_edge(u: int, v: int) -> None:
        """Append one directed edge ``u -> v`` and register it on both nodes."""
        n1, n2 = nodes[u], nodes[v]
        length = math.hypot(n2.x - n1.x, n2.y - n1.y)
        eid = len(edges)
        edges.append(Edge(id=eid, u=u, v=v, length=length))
        nodes[u].out_edges.append(eid)
        nodes[v].in_edges.append(eid)

    # Connect right and upward neighbours, each as a two-way pair.
    for j in range(height):
        for i in range(width):
            u = node_id[(i, j)]
            if i + 1 < width:
                v = node_id[(i + 1, j)]
                add_edge(u, v)
                add_edge(v, u)
            if j + 1 < height:
                v = node_id[(i, j + 1)]
                add_edge(u, v)
                add_edge(v, u)

    return RoadNetwork(nodes=nodes, edges=edges, node_id=node_id)


def build_city_grid(
    width: int,
    height: int,
    block: float = 150.0,
    *,
    seed: int = 0,
    jitter: float = 0.0,
    one_way_prob: float = 0.0,
    drop_prob: float = 0.0,
    arterial_every: int = 0,
    arterial_speed: float = 25.0,
    arterial_lanes: int = 2,
    ring: bool = False,
    ring_speed: float = 30.0,
    ring_lanes: int = 3,
    ring_access_spacing: float = 1000.0,
) -> RoadNetwork:
    """A heterogeneous grid: same ``(i, j)`` topology as :func:`build_grid_network`
    (so the H/V signal model still applies) but with cheap realism added.

    * ``jitter`` — node positions are randomly offset by up to ``jitter * block``
      in x and y (the grid indices ``i, j`` are unchanged, only ``x, y`` move).
    * ``one_way_prob`` — probability a neighbour connection is one-way (a single
      directed edge) instead of the usual two-way pair.
    * ``drop_prob`` — probability a neighbour connection is missing entirely (no
      edge at all), so blocks need not be fully connected. Arterial links are
      never dropped, which keeps the through-routes intact and the map mostly
      connected. Raises topological irregularity without breaking the ``(i, j)``
      indexing the signal model relies on.
    * ``arterial_every`` / ``arterial_speed`` / ``arterial_lanes`` — every
      ``arterial_every``-th row and column is an arterial with the higher
      ``arterial_speed`` limit and ``arterial_lanes`` lanes (default 2).
    * ``ring`` / ``ring_speed`` / ``ring_lanes`` — when ``ring=True`` the grid's
      perimeter (border rows/columns) becomes a fast ring road: ``ring_speed``
      limit and ``ring_lanes`` lanes (default 3). Ring beats arterial beats local.
      Everything else is a single-lane local street at ``DEFAULT_SPEED_LIMIT``.
      The ring is a **limited-access** road: on/off ramps (radials linking it to
      the interior) appear only about every ``ring_access_spacing`` metres, at
      least one per side. Elsewhere the ring runs past with no junction, so those
      border nodes see one orientation and aren't signalized — the ring flows
      freely, which keeps the fast ring physically coherent (a grade-separated
      ring is a later refinement). The ring loop is always two-way.
      (Lane counts populate ``Edge.lanes``.)

    ``block`` should be *physically coherent* with the speeds: a car must be able
    to stop within a block, i.e. ``block`` comfortably larger than the braking
    distance ``v^2 / (2 * comfortable_decel)``. At 50 km/h that is ~24 m and at
    90 km/h ~78 m, so the ~150 m default keeps even fast arterials stoppable
    before a signal; much shorter blocks put cars permanently in the dilemma zone
    (they cannot stop at a light within a block) and manufacture collisions.

    Two add-only repair passes run last, so the resulting network always has
    **no dead-ends** (every node can exit to at least two distinct neighbours, so
    no arrival is forced into a U-turn) and is **strongly connected** (every
    destination is reachable from everywhere). Only feasible on a genuine 2-D
    grid (``width``, ``height`` >= 2); a degenerate 1xN line still has endpoints.

    Seeded, so a given set of arguments always yields the same network.
    """
    rng = random.Random(seed)
    nodes: List[Node] = []
    node_id: Dict[Tuple[int, int], int] = {}

    nid = 0
    for j in range(height):
        for i in range(width):
            dx = rng.uniform(-jitter, jitter) * block
            dy = rng.uniform(-jitter, jitter) * block
            node_id[(i, j)] = nid
            nodes.append(Node(id=nid, i=i, j=j, x=i * block + dx, y=j * block + dy))
            nid += 1

    edges: List[Edge] = []

    def add_edge(u: int, v: int, speed: float, lanes: int = 1) -> None:
        """Append one directed edge ``u -> v`` with the given speed limit / lanes."""
        n1, n2 = nodes[u], nodes[v]
        length = math.hypot(n2.x - n1.x, n2.y - n1.y)
        eid = len(edges)
        edges.append(Edge(id=eid, u=u, v=v, length=length,
                          lanes=lanes, speed_limit=speed))
        nodes[u].out_edges.append(eid)
        nodes[v].in_edges.append(eid)

    def connect(u: int, v: int, speed: float, lanes: int, protected: bool,
                two_way: bool = False) -> None:
        """Connect neighbours ``u`` and ``v``, honouring ``drop_prob`` (skip the
        connection) and ``one_way_prob`` (a single directed edge instead of a
        two-way pair). ``protected`` connections (ring, arterial) are never
        dropped and never made one-way — an arterial corridor must carry both
        directions end to end; ``two_way`` ones (the ring loop) are likewise
        never one-way, so the loop stays strongly connected on its own."""
        if not protected and drop_prob and rng.random() < drop_prob:
            return
        if not two_way and one_way_prob and rng.random() < one_way_prob:
            # One-way: keep a single direction (chosen at random). The draws
            # happen for protected connections too (keeping the RNG stream —
            # and so the rest of the map — unchanged), but are ignored there.
            a, b = (u, v) if rng.random() < 0.5 else (v, u)
            if not protected:
                add_edge(a, b, speed, lanes)
                return
            add_edge(u, v, speed, lanes)
            add_edge(v, u, speed, lanes)
        else:
            add_edge(u, v, speed, lanes)
            add_edge(v, u, speed, lanes)

    def is_arterial(index: int) -> bool:
        """Whether grid row/column ``index`` is an arterial (higher speed)."""
        return arterial_every > 0 and index % arterial_every == 0

    def classify(on_border: bool, index: int):
        """Return ``(speed, lanes, protected)`` for a connection. A perimeter
        (ring) connection wins over arterial, which wins over local."""
        if ring and on_border:
            return ring_speed, ring_lanes, True
        if is_arterial(index):
            return arterial_speed, arterial_lanes, True
        return DEFAULT_SPEED_LIMIT, 1, False

    # Limited-access ring: on/off ramps (radials linking the ring to the
    # interior) only at a few interchange rows/columns — about one per
    # ``ring_access_spacing`` metres, at least one per side. Elsewhere the ring
    # runs past with no junction (those border nodes see only one orientation, so
    # they are unsignalized and the ring flows freely — which also keeps the fast
    # ring physically coherent). A grade-separated ring is a later refinement.
    spacing = max(1.0, ring_access_spacing / block)
    access_rows = _ring_interchanges(height, spacing) if ring else set()
    access_cols = _ring_interchanges(width, spacing) if ring else set()

    for j in range(height):
        for i in range(width):
            u = node_id[(i, j)]
            if i + 1 < width:  # horizontal connection lies on row j
                on_border = (j == 0 or j == height - 1)
                radial = ring and not on_border and (i == 0 or i + 1 == width - 1)
                ramp = radial and j in access_rows          # kept on/off ramp
                if radial and not ramp:
                    pass  # no ring on/off ramp here
                else:
                    speed, lanes, prot = classify(on_border, j)
                    connect(u, node_id[(i + 1, j)], speed, lanes, prot or ramp,
                            two_way=on_border or ramp)
            if j + 1 < height:  # vertical connection lies on column i
                on_border = (i == 0 or i == width - 1)
                radial = ring and not on_border and (j == 0 or j + 1 == height - 1)
                ramp = radial and i in access_cols
                if radial and not ramp:
                    pass  # no ring on/off ramp here
                else:
                    speed, lanes, prot = classify(on_border, i)
                    connect(u, node_id[(i, j + 1)], speed, lanes, prot or ramp,
                            two_way=on_border or ramp)

    def grid_neighbours(node: Node):
        """Yield the node ids of ``node``'s existing grid neighbours (E/W/N/S)."""
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = node_id.get((node.i + di, node.j + dj))
            if nbr is not None:
                yield nbr

    # Two repair passes, both add-only (so the strong-connectivity guarantee
    # established last survives the earlier one):
    #  1. No dead-ends: guarantee every node can exit to >= 2 distinct nodes, so
    #     an arriving car is never forced into a U-turn (drops / one-ways can
    #     otherwise leave a node hanging off the grid by a single street).
    #  2. Strong connectivity: dropping edges can still fragment the grid into
    #     pockets; bridge grid-adjacent component boundaries two-way until the
    #     graph is a single component, so any car can reach any destination.
    _ensure_min_exit_degree(nodes, edges, grid_neighbours, add_edge)
    _make_strongly_connected(nodes, edges, grid_neighbours, add_edge)

    return RoadNetwork(nodes=nodes, edges=edges, node_id=node_id)


def _ring_interchanges(dim: int, spacing: float) -> set:
    """Interior indices along a side of ``dim`` nodes that are ring interchanges.

    About one every ``spacing`` blocks, at least one, evenly spaced among the
    interior positions (corners are always on the ring, never on/off ramps).
    """
    interior = list(range(1, dim - 1))
    if not interior:
        return set()
    count = min(len(interior), max(1, round((dim - 1) / spacing)))
    if count == 1:
        return {interior[len(interior) // 2]}
    step = (len(interior) - 1) / (count - 1)
    return {interior[round(k * step)] for k in range(count)}


def _strongly_connected_components(nodes: List[Node], edges: List[Edge]) -> List[int]:
    """Component id per node via Kosaraju's algorithm (iterative)."""
    n = len(nodes)
    order: List[int] = []
    seen = [False] * n
    # First pass: finish-time order on the forward graph.
    for start in range(n):
        if seen[start]:
            continue
        stack = [(start, iter(nodes[start].out_edges))]
        seen[start] = True
        while stack:
            node, it = stack[-1]
            for eid in it:
                w = edges[eid].v
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, iter(nodes[w].out_edges)))
                    break
            else:
                order.append(node)
                stack.pop()
    # Second pass: DFS on the reverse graph in reverse finish order.
    comp = [-1] * n
    cid = 0
    for start in reversed(order):
        if comp[start] != -1:
            continue
        stack = [start]
        comp[start] = cid
        while stack:
            node = stack.pop()
            for eid in nodes[node].in_edges:  # reverse edges u -> node
                u = edges[eid].u
                if comp[u] == -1:
                    comp[u] = cid
                    stack.append(u)
        cid += 1
    return comp


def _ensure_min_exit_degree(nodes, edges, grid_neighbours, add_edge,
                            min_neighbours: int = 2) -> None:
    """Guarantee every node can exit to at least ``min_neighbours`` *distinct*
    nodes, so no arrival is ever forced into a U-turn (a "dead-end street").

    For each node short of the target, add two-way links to grid-adjacent
    neighbours it does not already exit to, until it reaches the target or runs
    out of grid neighbours (a node with fewer grid neighbours than the target —
    only possible on a degenerate 1xN grid — is connected to all it has). One
    forward pass suffices: edges are only ever added, so once a node meets the
    target it stays there, and links added for a neighbour only help it too.
    """
    for node in nodes:
        exits = {edges[eid].v for eid in node.out_edges}
        neighbours = list(grid_neighbours(node))
        target = min(min_neighbours, len(neighbours))
        for nbr in neighbours:
            if len(exits) >= target:
                break
            if nbr in exits:
                continue
            add_edge(node.id, nbr, DEFAULT_SPEED_LIMIT)
            add_edge(nbr, node.id, DEFAULT_SPEED_LIMIT)
            exits.add(nbr)


def _make_strongly_connected(nodes, edges, grid_neighbours, add_edge) -> None:
    """Add two-way edges across grid-adjacent component boundaries until the
    directed graph is a single strongly connected component."""
    while True:
        comp = _strongly_connected_components(nodes, edges)
        if max(comp) == 0:  # one component
            return
        # Find the first grid-adjacent pair in different components and bridge it.
        for node in nodes:
            for nbr in grid_neighbours(node):
                if comp[node.id] != comp[nbr]:
                    add_edge(node.id, nbr, DEFAULT_SPEED_LIMIT)
                    add_edge(nbr, node.id, DEFAULT_SPEED_LIMIT)
                    break
            else:
                continue
            break
