"""Routing policies: deciding which edge a car takes at an intersection.

Two pluggable routers share one interface, ``next_edge(edge_id, car)``:

* :class:`RandomRouter` — memoryless wandering (ignores the car's destination).
* :class:`ShortestPathRouter` — steers each car toward ``car.dest`` along the
  fastest free-flow path.

Both are injected into :class:`~traffic_sim.simulation.TrafficSim` the same way,
so swapping routing strategy never touches the simulation core.
"""

import heapq
import math
import random
from typing import Dict, Optional

from .network import RoadNetwork
from .vehicles import Car


class RandomRouter:
    """Pick the next edge uniformly at random among a node's out-edges.

    U-turns (immediately going back down the edge just travelled) are avoided
    unless the node is a dead-end with no other option. Seeded for
    deterministic, reproducible runs. The ``car`` argument is accepted for
    interface compatibility but ignored — this router has no destination.

    Turns are permissive: the simulation models no intersection right-of-way,
    so a chosen turn may cross oncoming traffic sharing the same green phase.
    Protected turns / gap acceptance are a deferred future phase.
    """

    def __init__(self, net: RoadNetwork, seed: int = 0, allow_uturn: bool = False) -> None:
        """``seed`` fixes the random choices (reproducible runs); set
        ``allow_uturn`` to permit reversing back down the approach edge."""
        self.net = net
        self.rng = random.Random(seed)
        self.allow_uturn = allow_uturn

    def next_edge(self, edge_id: int, car: Optional[Car] = None) -> Optional[int]:
        """Pick a random out-edge at the end of ``edge_id`` (``car`` ignored).

        Returns ``None`` at a dead-end with no outgoing edge; otherwise avoids
        the U-turn back onto ``edge_id`` unless that is the only option.
        """
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


class ShortestPathRouter:
    """Route each car toward its destination node along the fastest path.

    Edge cost is free-flow travel time (``length / speed_limit``), so cars
    prefer faster arterials over slower local streets. The cost-to-go from
    every node to a given destination is computed once with Dijkstra over the
    reversed graph and cached (the network is static), then at each intersection
    the chosen out-edge minimises ``edge_cost + cost_to_go(neighbour)``.

    A car with no destination (or one that has just reached its destination) is
    given a fresh random destination, so the simulation keeps flowing and
    density stays roughly constant — useful as a steady-state training signal.
    If the destination happens to be unreachable from the current node (e.g.
    behind one-way streets), the car falls back to a random non-U-turn move and
    re-routes next step. Seeded for determinism.
    """

    def __init__(self, net: RoadNetwork, seed: int = 0) -> None:
        """``seed`` fixes the random destination assignments and the
        tie-break used when a destination is unreachable, for reproducible runs.
        Per-destination cost tables are computed lazily and cached here."""
        self.net = net
        self.rng = random.Random(seed)
        # dest node id -> {node id: free-flow cost to reach dest}
        self._dist_cache: Dict[int, Dict[int, float]] = {}

    @staticmethod
    def _cost(edge) -> float:
        """Free-flow travel time along an edge [s]."""
        return edge.length / edge.speed_limit if edge.speed_limit > 0 else math.inf

    def _dist_to(self, dest: int) -> Dict[int, float]:
        """Min free-flow cost from every node to ``dest`` (Dijkstra, reversed)."""
        cached = self._dist_cache.get(dest)
        if cached is not None:
            return cached

        dist: Dict[int, float] = {dest: 0.0}
        pq = [(0.0, dest)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist.get(v, math.inf):
                continue
            # Walk edges u -> v backwards: reaching v lets us reach u.
            for eid in self.net.nodes[v].in_edges:
                e = self.net.edges[eid]
                nd = d + self._cost(e)
                if nd < dist.get(e.u, math.inf):
                    dist[e.u] = nd
                    heapq.heappush(pq, (nd, e.u))

        self._dist_cache[dest] = dist
        return dist

    def free_flow_time(self, origin: int, dest: int) -> float:
        """Ideal (uncongested) travel time [s] from ``origin`` node to ``dest``.

        The same cost-to-go used for routing — the fastest free-flow path, summing
        each edge's ``length / speed_limit``. Metrics use it as the baseline a
        trip's delay is measured against. Returns ``inf`` if ``dest`` is
        unreachable from ``origin``.
        """
        return self._dist_to(dest).get(origin, math.inf)

    def assign_destination(self, car: Car, avoid: Optional[int] = None) -> int:
        """Give ``car`` a new random destination node (never ``avoid``)."""
        n = len(self.net.nodes)
        dest = self.rng.randrange(n)
        if avoid is not None and n > 1:
            while dest == avoid:
                dest = self.rng.randrange(n)
        car.dest = dest
        return dest

    def next_edge(self, edge_id: int, car: Optional[Car] = None) -> Optional[int]:
        """Choose the out-edge at the end of ``edge_id`` that best serves ``car.dest``.

        ``car`` is required (this router steers by its ``dest``); passing ``None``
        raises. If the car has no destination or has just reached it, a fresh one
        is assigned. Returns the out-edge minimising ``edge_cost +
        cost_to_go(neighbour)``, ``None`` at a dead-end, or — when the
        destination is unreachable from here — a random fallback so the car keeps
        moving and re-routes next step.

        U-turns (reversing straight back down the approach edge) are forbidden:
        the reverse edge is excluded from consideration unless the node is a
        genuine dead-end where it is the only way out. (A roundabout would be the
        other exemption, once node control types model one.)
        """
        if car is None:
            raise ValueError("ShortestPathRouter requires a car (it routes to car.dest)")

        edge = self.net.edges[edge_id]
        node = edge.v  # the intersection the car is approaching
        options = self.net.nodes[node].out_edges
        if not options:
            return None  # dead-end

        # Forbid U-turns: drop the edge that leads straight back to where we came
        # from, unless it is the only option (a true dead-end).
        forward = [eid for eid in options if self.net.edges[eid].v != edge.u]
        candidates = forward or list(options)

        # No destination yet, or we've just arrived at it: pick a fresh one.
        if car.dest is None or car.dest == node:
            self.assign_destination(car, avoid=node)

        dist = self._dist_to(car.dest)
        best, best_cost = None, math.inf
        for eid in candidates:
            e = self.net.edges[eid]
            cost = self._cost(e) + dist.get(e.v, math.inf)
            if cost < best_cost:
                best_cost, best = cost, eid

        if best is None or best_cost == math.inf:
            # Destination unreachable without a U-turn: wander forward and
            # re-route next step once we're somewhere with a path.
            return self.rng.choice(candidates)
        return best
