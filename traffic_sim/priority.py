"""Right-of-way at unsignalized intersections: a priority + gap-acceptance gate.

Signalized nodes are handled by :mod:`signals`; this fills the documented gap at
*unsignalized* nodes, where conflicting movements would otherwise pass through
each other. Pluggable like the router and signal controller — inject a
:class:`PriorityModel` into :class:`~traffic_sim.simulation.TrafficSim` and it
gates entry to unsignalized nodes (``priority=None`` keeps the old free-for-all).

Model (deliberately simple and readable): the front car on each approach may
enter the node unless a *conflicting* movement from a *higher-priority* approach
is *imminent*.

* **Priority is a total order** — arterials (edges whose ``speed_limit`` exceeds
  the default) outrank local streets; ties are broken deterministically by edge
  id. A total order means contests always resolve and never deadlock (unlike a
  pure yield-to-the-right rule, which can cycle at a 4-way).
* **Imminent = gap acceptance** — the higher-priority car is within a critical
  distance that grows with its speed (``max(min_gap, speed * critical_gap)``). A
  higher-priority car stopped far away therefore does *not* block (avoids
  freezing), but one at the stop line does.
* **Right turns are exempt** — they stay in the nearest lane, so they neither
  yield nor block.

Yielding reuses the simulation's existing red-stop-line obstacle, so no new
dynamics are introduced. Known simplifications (candidate refinements): the
edge-id tie-break is arbitrary rather than yield-to-the-right; only approach
*fronts* contest (a car already mid-crossing is not re-checked); lanes/multiple
gaps are not modelled.
"""

from typing import List, Optional, Tuple

from .network import RoadNetwork, DEFAULT_SPEED_LIMIT
from .signals import edge_orientation, turn_type, TurnType

# One contender at a node: (from_edge, to_edge, gap_to_node [m], speed [m/s]).
Contender = Tuple[int, Optional[int], float, float]


class PriorityModel:
    """Gap-acceptance right-of-way for unsignalized nodes (see module docs).

    Stateless apart from its tuning parameters and a reference to the network;
    the simulation gathers the contending cars each step and asks
    :meth:`must_yield` whether an approach's front car has to give way.
    """

    def __init__(self, net: RoadNetwork, *, trigger_dist: float = 30.0,
                 critical_gap: float = 2.5, min_gap: float = 6.0) -> None:
        """Configure the right-of-way thresholds.

        ``trigger_dist`` — only cars within this distance of a node are
        considered contenders [m]. ``critical_gap`` — the critical time headway
        used for gap acceptance; a higher-priority car counts as "imminent" when
        it is within ``speed * critical_gap`` [s]. ``min_gap`` — a floor on that
        distance, so a car essentially at the stop line always counts as present
        even at low speed [m].
        """
        self.net = net
        self.trigger_dist = trigger_dist  # only cars this near a node contest [m]
        self.critical_gap = critical_gap  # critical time headway for acceptance [s]
        self.min_gap = min_gap            # closer than this counts as "at the node" [m]

    def _is_arterial(self, edge_id: int) -> bool:
        """Whether ``edge_id`` is an arterial (speed limit above the default)."""
        return self.net.edges[edge_id].speed_limit > DEFAULT_SPEED_LIMIT + 1e-6

    def _rank(self, edge_id: int) -> Tuple[int, int]:
        """Total-order priority key for an approach (higher = more priority)."""
        return (1 if self._is_arterial(edge_id) else 0, edge_id)

    def _conflict(self, from_a: int, to_a: int, from_b: int, to_b: int) -> bool:
        """Whether two movements crossing the same node conflict."""
        if turn_type(self.net, from_a, to_a) is TurnType.RIGHT:
            return False
        if turn_type(self.net, from_b, to_b) is TurnType.RIGHT:
            return False
        # Two straight movements on the same axis run parallel (they enter from
        # opposite ends), so they do not cross.
        oa, ob = edge_orientation(self.net, from_a), edge_orientation(self.net, from_b)
        sa = turn_type(self.net, from_a, to_a) is TurnType.STRAIGHT
        sb = turn_type(self.net, from_b, to_b) is TurnType.STRAIGHT
        if oa is ob and sa and sb:
            return False
        return True

    def _imminent(self, gap: float, speed: float) -> bool:
        """Whether a contender ``gap`` metres away at ``speed`` will enter soon.

        This is the gap-acceptance test: the horizon grows with speed
        (``speed * critical_gap``) but never falls below ``min_gap``, so a fast
        approaching car blocks from farther out while a car already at the line
        blocks regardless of speed.
        """
        return gap <= max(self.min_gap, speed * self.critical_gap)

    def must_yield(self, my_from: int, my_to: Optional[int],
                   contenders: List[Contender]) -> bool:
        """True if the front car on ``my_from`` (committed to ``my_to``) must
        yield to a higher-priority conflicting contender at the node."""
        if my_to is None:
            return False  # dead-end: nothing to enter
        if turn_type(self.net, my_from, my_to) is TurnType.RIGHT:
            return False  # right turns never yield in this model
        my_rank = self._rank(my_from)
        for from_e, to_e, gap, speed in contenders:
            if from_e == my_from or to_e is None:
                continue
            if self._rank(from_e) <= my_rank:
                continue  # not higher priority
            if not self._conflict(my_from, my_to, from_e, to_e):
                continue
            if self._imminent(gap, speed):
                return True
        return False
