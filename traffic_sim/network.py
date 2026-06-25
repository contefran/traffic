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


@dataclass
class Node:
    id: int
    i: int  # grid column
    j: int  # grid row
    x: float
    y: float
    out_edges: List[int] = field(default_factory=list)
    in_edges: List[int] = field(default_factory=list)


@dataclass
class Edge:
    id: int
    u: int  # source node id
    v: int  # target node id
    length: float
    lanes: int = 1
    speed_limit: float = DEFAULT_SPEED_LIMIT  # [m/s]


@dataclass
class RoadNetwork:
    nodes: List[Node]
    edges: List[Edge]
    node_id: Dict[Tuple[int, int], int]  # (i, j) -> node id

    def point_on_edge(self, edge_id: int, s: float) -> Tuple[float, float]:
        """World (x, y) of a point ``s`` metres along ``edge_id``."""
        e = self.edges[edge_id]
        n1, n2 = self.nodes[e.u], self.nodes[e.v]
        t = s / e.length if e.length > 0 else 0.0
        return (n1.x + t * (n2.x - n1.x), n1.y + t * (n2.y - n1.y))

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
    block: float = 50.0,
    *,
    seed: int = 0,
    jitter: float = 0.0,
    one_way_prob: float = 0.0,
    arterial_every: int = 0,
    arterial_speed: float = 25.0,
) -> RoadNetwork:
    """A heterogeneous grid: same ``(i, j)`` topology as :func:`build_grid_network`
    (so the H/V signal model still applies) but with cheap realism added.

    * ``jitter`` — node positions are randomly offset by up to ``jitter * block``
      in x and y (the grid indices ``i, j`` are unchanged, only ``x, y`` move).
    * ``one_way_prob`` — probability a neighbour connection is one-way (a single
      directed edge) instead of the usual two-way pair.
    * ``arterial_every`` / ``arterial_speed`` — every ``arterial_every``-th row and
      column is an arterial whose edges get the higher ``arterial_speed`` limit.

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

    def add_edge(u: int, v: int, speed: float) -> None:
        n1, n2 = nodes[u], nodes[v]
        length = math.hypot(n2.x - n1.x, n2.y - n1.y)
        eid = len(edges)
        edges.append(Edge(id=eid, u=u, v=v, length=length, speed_limit=speed))
        nodes[u].out_edges.append(eid)
        nodes[v].in_edges.append(eid)

    def connect(u: int, v: int, speed: float) -> None:
        if one_way_prob and rng.random() < one_way_prob:
            # One-way: keep a single direction (chosen at random).
            a, b = (u, v) if rng.random() < 0.5 else (v, u)
            add_edge(a, b, speed)
        else:
            add_edge(u, v, speed)
            add_edge(v, u, speed)

    def is_arterial(index: int) -> bool:
        return arterial_every > 0 and index % arterial_every == 0

    for j in range(height):
        for i in range(width):
            u = node_id[(i, j)]
            if i + 1 < width:  # horizontal connection lies on row j
                speed = arterial_speed if is_arterial(j) else DEFAULT_SPEED_LIMIT
                connect(u, node_id[(i + 1, j)], speed)
            if j + 1 < height:  # vertical connection lies on column i
                speed = arterial_speed if is_arterial(i) else DEFAULT_SPEED_LIMIT
                connect(u, node_id[(i, j + 1)], speed)

    return RoadNetwork(nodes=nodes, edges=edges, node_id=node_id)
