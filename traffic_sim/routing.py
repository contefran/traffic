"""Routing policies: deciding which edge a car takes at an intersection."""

import random
from typing import Optional

from .network import RoadNetwork


class RandomRouter:
    """Pick the next edge uniformly at random among a node's out-edges.

    U-turns (immediately going back down the edge just travelled) are avoided
    unless the node is a dead-end with no other option. Seeded for
    deterministic, reproducible runs.
    """

    def __init__(self, net: RoadNetwork, seed: int = 0, allow_uturn: bool = False) -> None:
        self.net = net
        self.rng = random.Random(seed)
        self.allow_uturn = allow_uturn

    def next_edge(self, edge_id: int) -> Optional[int]:
        edge = self.net.edges[edge_id]
        node = self.net.nodes[edge.v]
        options = list(node.out_edges)
        if not options:
            return None  # dead-end with no outgoing edges
        if not self.allow_uturn:
            # The U-turn is the out-edge leading back to where we came from.
            forward = [eid for eid in options if self.net.edges[eid].v != edge.u]
            options = forward or options
        return self.rng.choice(options)
