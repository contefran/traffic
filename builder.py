from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math


@dataclass
class Node:
    id: int
    i: int
    j: int
    x: float
    y: float
    out_edges: List[int] = field(default_factory=list)
    in_edges: List[int] = field(default_factory=list)

@dataclass
class Edge:
    id: int
    u: int
    v: int
    length: float
    lanes: int = 1
    speed_limit: float = 13.9 # ¬50 km/h

@dataclass
class RoadNetwork:
    nodes: List[Node]
    edges: List[Edge]
    node_id: Dict[Tuple[int, int], int]  # (i,j) -> node id


class Builder:

    def __init__(self, W: int, H: int, block: float) -> None:
        self.W = W
        self.H = H
        self.block = block

    def build_grid_network(self) -> RoadNetwork:
        nodes: List[Node] = []
        node_id: Dict[Tuple[int, int], int] = {}

        # 1) create nodes
        nid = 0
        for j in range(self.H):
            for i in range(self.W):
                x, y = i * self.block, j * self.block
                node_id[(i, j)] = nid
                nodes.append(Node(id=nid, i=i, j=j, x=x, y=y))
                nid += 1

        edges: List[Edge] = []
        eid = 0

        def add_edge(u: int, v: int):
            nonlocal eid
            n1, n2 = nodes[u], nodes[v]
            length = math.hypot(n2.x - n1.x, n2.y - n1.y)
            edges.append(Edge(id=eid, u=u, v=v, length=length))
            nodes[u].out_edges.append(eid)
            nodes[v].in_edges.append(eid)
            eid += 1

        # 2) connect neighbours (two-way)
        for j in range(self.H):
            for i in range(self.W):
                u = node_id[(i, j)]
                if i + 1 < self.W:
                    v = node_id[(i + 1, j)]
                    add_edge(u, v)
                    add_edge(v, u)
                if j + 1 < self.H:
                    v = node_id[(i, j + 1)]
                    add_edge(u, v)
                    add_edge(v, u)

        return RoadNetwork(nodes=nodes, edges=edges, node_id=node_id)
