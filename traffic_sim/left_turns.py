"""Permissive-left gap acceptance at signalized intersections.

Under *permissive* phasing (:class:`~traffic_sim.signals.FixedTimeController`) a
left turn shares its green with the oncoming straight-through stream on the same
axis, so a realistic left-turner must **yield** — wait in the intersection for an
acceptable **gap** in that oncoming traffic before turning. Without this a
permissive left is unrealistically free (see Q18).

Pluggable and opt-in, like :class:`~traffic_sim.priority.PriorityModel`: inject a
:class:`PermissiveLeftModel` into :class:`~traffic_sim.simulation.TrafficSim`
(``left_turn=None`` keeps the old free-left behaviour). The model is **inert
under protected phasing**: when a protected left has green, the opposing through
is red, so there is nothing to yield to — the same object is correct for both
controllers and only bites under permissive.

Simplifications: only the *front* car of an approach is gated (single lane, so
the ones behind are held by car-following anyway); the oncoming front car
conflicts regardless of its own turn (a mild over-yield); gap acceptance reuses
the same ``gap <= max(min_gap, speed * critical_gap)`` test as the priority model.
"""

from typing import Dict, Optional, Tuple

from .network import RoadNetwork
from .signals import turn_type, TurnType, SignalState


class PermissiveLeftModel:
    """Gap-acceptance yield for permissive left turns (see module docs)."""

    def __init__(self, net: RoadNetwork, *, critical_gap: float = 2.5,
                 min_gap: float = 6.0) -> None:
        """Configure the gap thresholds and precompute the opposing-approach map.

        ``critical_gap`` [s] and ``min_gap`` [m] mirror the priority model: a
        left-turner accepts a gap only when the nearest oncoming car is farther
        than ``max(min_gap, speed * critical_gap)``. Smaller values = more
        aggressive lefts (fewer, shorter waits).
        """
        self.net = net
        self.critical_gap = critical_gap
        self.min_gap = min_gap
        # approach edge id -> (opposing approach edge, its straight-through exit)
        self._opposing: Dict[int, Tuple[int, int]] = self._build_opposing()

    def _edge(self, u: int, v: int) -> Optional[int]:
        """The edge id ``u -> v`` if it exists, else ``None``."""
        for eid in self.net.nodes[u].out_edges:
            if self.net.edges[eid].v == v:
                return eid
        return None

    def _build_opposing(self) -> Dict[int, Tuple[int, int]]:
        """For each approach ``U -> X``, find the opposing approach ``O -> X`` (O
        grid-collinear with U through X) and its straight exit ``X -> U``.

        Both must exist for an oncoming-through conflict to be possible; entries
        without a full opposing through are simply omitted. Uses grid indices, so
        it is unaffected by positional jitter.
        """
        opp: Dict[int, Tuple[int, int]] = {}
        nid = self.net.node_id
        for e in self.net.edges:
            x, u = self.net.nodes[e.v], self.net.nodes[e.u]
            o = nid.get((2 * x.i - u.i, 2 * x.j - u.j))   # opposite side of X from U
            if o is None:
                continue
            opposing_edge = self._edge(o, e.v)            # O -> X
            straight_exit = self._edge(e.v, e.u)          # X -> U (opposing through)
            if opposing_edge is not None and straight_exit is not None:
                opp[e.id] = (opposing_edge, straight_exit)
        return opp

    def must_yield(self, from_edge: int, to_edge: int, signals, t: float,
                   cars_on_edge: Dict[int, list]) -> bool:
        """True if the front left-turner ``from_edge -> to_edge`` must wait for a
        gap in oncoming through traffic.

        ``cars_on_edge`` is the simulation's per-edge lists, sorted front-first
        (highest ``s``). Returns ``False`` for non-left movements, when there is
        no opposing through, when the opposing through is red (protected phasing —
        no conflict), or when the nearest oncoming car is not imminent.
        """
        if signals is None:
            return False
        if turn_type(self.net, from_edge, to_edge) not in (TurnType.LEFT, TurnType.UTURN):
            return False
        opp = self._opposing.get(from_edge)
        if opp is None:
            return False
        opposing_edge, straight_exit = opp
        # Only a conflict while the oncoming through actually has right of way.
        if signals.movement_state(opposing_edge, straight_exit, t) is SignalState.RED:
            return False
        lst = cars_on_edge.get(opposing_edge)
        if not lst:
            return False
        front = lst[0]                                    # nearest the node
        gap = self.net.edges[opposing_edge].length - front.s
        return gap <= max(self.min_gap, front.v * self.critical_gap)
