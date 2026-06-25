"""Traffic signals: which approaches to an intersection may pass right now.

The model is deliberately pluggable, mirroring ``RandomRouter``:

* A :class:`SignalController` decides, per node and time, which road
  *orientation* (horizontal vs vertical) currently has a green light. Swap in
  an adaptive or learned controller later without touching the simulation.
* :class:`SignalSystem` wraps a controller and answers the only question the
  simulation needs: ``is_green(edge_id, t)`` for a car about to leave that edge.

Nodes that carry only one orientation (grid borders, dead-ends) are never
signalized — there is no cross traffic to protect, so they always pass.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Protocol

from .network import RoadNetwork


class Orientation(Enum):
    HORIZONTAL = "H"  # edge runs east-west (endpoints share a grid row j)
    VERTICAL = "V"    # edge runs north-south (endpoints share a grid column i)


def edge_orientation(net: RoadNetwork, edge_id: int) -> Orientation:
    e = net.edges[edge_id]
    u, v = net.nodes[e.u], net.nodes[e.v]
    return Orientation.HORIZONTAL if u.j == v.j else Orientation.VERTICAL


class SignalController(Protocol):
    """Returns the orientation that has green at ``node_id`` at time ``t``."""

    def green_orientation(self, node_id: int, t: float) -> Orientation:
        ...


@dataclass
class FixedTimeController:
    """Every intersection alternates on a fixed timer, all in phase.

    ``node_id`` is accepted but unused here; per-node controllers (green waves,
    adaptive timing) can use it.
    """

    green_time: float = 10.0
    start: Orientation = Orientation.HORIZONTAL

    def green_orientation(self, node_id: int, t: float) -> Orientation:
        flipped = int(t // self.green_time) % 2 == 1
        other = (Orientation.VERTICAL if self.start is Orientation.HORIZONTAL
                 else Orientation.HORIZONTAL)
        return other if flipped else self.start


class SignalSystem:
    def __init__(self, net: RoadNetwork, controller: SignalController) -> None:
        self.net = net
        self.controller = controller
        self._orientation: Dict[int, Orientation] = {
            e.id: edge_orientation(net, e.id) for e in net.edges
        }
        # A node is signalized only if it mixes both orientations.
        self._signalized: Dict[int, bool] = {}
        for node in net.nodes:
            orients = {self._orientation[eid] for eid in node.in_edges}
            self._signalized[node.id] = (
                Orientation.HORIZONTAL in orients and Orientation.VERTICAL in orients
            )

    def is_signalized(self, node_id: int) -> bool:
        return self._signalized[node_id]

    def is_green(self, edge_id: int, t: float) -> bool:
        """True if a car may leave ``edge_id`` into its downstream node at ``t``."""
        node_id = self.net.edges[edge_id].v
        if not self._signalized[node_id]:
            return True
        return self._orientation[edge_id] is self.controller.green_orientation(node_id, t)
