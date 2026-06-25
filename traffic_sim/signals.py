"""Traffic signals: which *movements* through an intersection may proceed now.

Pluggable, mirroring ``RandomRouter``:

* A :class:`SignalController` decides, per node/time, whether a given movement
  (an approach *orientation* + a *turn* type) is allowed. Swap in an adaptive or
  learned controller without touching the simulation.
* :class:`SignalSystem` wraps a controller and answers the question the
  simulation asks: ``allows_movement(from_edge, to_edge, t)`` for a car about to
  cross. ``is_green(edge_id, t)`` (proceed straight) is kept for convenience.

Two controllers ship: :class:`FixedTimeController` (the simple 2-phase
orientation light — *permissive* turns, left turns may conflict with oncoming
through traffic) and :class:`ProtectedPhaseController` (a 4-phase cycle giving
left turns their own protected phase).

Nodes carrying only one orientation (grid borders) are never signalized; a node
may also be forced unsignalized via ``SignalSystem(unsignalized_nodes=...)``.
NOTE: an unsignalized node currently has **no right-of-way** modelling — all
movements are permitted, so conflicting traffic passes through unchecked. That
is a known, deliberate gap.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Protocol, Set

from .network import RoadNetwork


class Orientation(Enum):
    HORIZONTAL = "H"  # edge runs east-west (endpoints share a grid row j)
    VERTICAL = "V"    # edge runs north-south (endpoints share a grid column i)


class TurnType(Enum):
    STRAIGHT = "S"
    LEFT = "L"
    RIGHT = "R"
    UTURN = "U"


def edge_orientation(net: RoadNetwork, edge_id: int) -> Orientation:
    e = net.edges[edge_id]
    u, v = net.nodes[e.u], net.nodes[e.v]
    return Orientation.HORIZONTAL if u.j == v.j else Orientation.VERTICAL


def turn_type(net: RoadNetwork, from_edge: int, to_edge: int) -> TurnType:
    """Classify the from_edge -> to_edge movement from the actual headings.

    Uses the signed angle between the approach heading and the exit heading, so
    it is correct even on jittered / non-axis-aligned geometry. In standard
    (x-right, y-up) coordinates a positive (counter-clockwise) turn is a left.
    """
    e_in, e_out = net.edges[from_edge], net.edges[to_edge]
    ix = net.nodes[e_in.v].x - net.nodes[e_in.u].x
    iy = net.nodes[e_in.v].y - net.nodes[e_in.u].y
    ox = net.nodes[e_out.v].x - net.nodes[e_out.u].x
    oy = net.nodes[e_out.v].y - net.nodes[e_out.u].y
    cross = ix * oy - iy * ox
    dot = ix * ox + iy * oy
    angle = math.degrees(math.atan2(cross, dot))
    if abs(angle) < 45.0:
        return TurnType.STRAIGHT
    if 45.0 <= angle < 135.0:
        return TurnType.LEFT
    if -135.0 < angle <= -45.0:
        return TurnType.RIGHT
    return TurnType.UTURN


class SignalController(Protocol):
    """Whether a movement (approach ``orientation`` + ``turn``) is allowed."""

    def allows(self, node_id: int, orientation: Orientation,
               turn: TurnType, t: float) -> bool:
        ...


@dataclass
class FixedTimeController:
    """Simple 2-phase light: each intersection alternates H/V on a fixed timer.

    Turns are *permissive* — any turn is allowed whenever your approach
    orientation has green, so a left turn can conflict with oncoming through
    traffic. ``node_id`` is accepted but unused (all nodes run in phase).
    """

    green_time: float = 10.0
    start: Orientation = Orientation.HORIZONTAL

    def green_orientation(self, node_id: int, t: float) -> Orientation:
        flipped = int(t // self.green_time) % 2 == 1
        other = (Orientation.VERTICAL if self.start is Orientation.HORIZONTAL
                 else Orientation.HORIZONTAL)
        return other if flipped else self.start

    def allows(self, node_id: int, orientation: Orientation,
               turn: TurnType, t: float) -> bool:
        return orientation is self.green_orientation(node_id, t)


@dataclass
class ProtectedPhaseController:
    """4-phase cycle giving left turns a protected phase, all nodes in phase.

    Phase order (each lasts ``green_time``):
      0  H through + right
      1  H left (and U-turn)
      2  V through + right
      3  V left (and U-turn)

    Because through and left never share a phase, a protected left never crosses
    oncoming through traffic.
    """

    green_time: float = 5.0

    _LABELS = ("H→", "H↰", "V→", "V↰")

    def phase(self, t: float) -> int:
        return int(t // self.green_time) % 4

    def phase_label(self, t: float) -> str:
        return self._LABELS[self.phase(t)]

    def allows(self, node_id: int, orientation: Orientation,
               turn: TurnType, t: float) -> bool:
        k = self.phase(t)
        is_left = turn in (TurnType.LEFT, TurnType.UTURN)
        if orientation is Orientation.HORIZONTAL:
            return k == (1 if is_left else 0)
        return k == (3 if is_left else 2)


class SignalSystem:
    def __init__(self, net: RoadNetwork, controller: SignalController,
                 unsignalized_nodes: Optional[Set[int]] = None) -> None:
        self.net = net
        self.controller = controller
        forced_off = unsignalized_nodes or set()
        self._orientation: Dict[int, Orientation] = {
            e.id: edge_orientation(net, e.id) for e in net.edges
        }
        # Signalized only if the node mixes both orientations and isn't forced off.
        self._signalized: Dict[int, bool] = {}
        for node in net.nodes:
            orients = {self._orientation[eid] for eid in node.in_edges}
            both = (Orientation.HORIZONTAL in orients
                    and Orientation.VERTICAL in orients)
            self._signalized[node.id] = both and node.id not in forced_off

    def is_signalized(self, node_id: int) -> bool:
        return self._signalized[node_id]

    def allows(self, node_id: int, orientation: Orientation,
               turn: TurnType, t: float) -> bool:
        """Node-level movement query (used for rendering)."""
        if not self._signalized[node_id]:
            return True
        return self.controller.allows(node_id, orientation, turn, t)

    def allows_movement(self, from_edge: int, to_edge: int, t: float) -> bool:
        """True if a car may cross from ``from_edge`` onto ``to_edge`` at ``t``."""
        node_id = self.net.edges[from_edge].v
        if not self._signalized[node_id]:
            return True
        orientation = self._orientation[from_edge]
        turn = turn_type(self.net, from_edge, to_edge)
        return self.controller.allows(node_id, orientation, turn, t)

    def is_green(self, edge_id: int, t: float) -> bool:
        """Convenience: may a car proceed *straight* off ``edge_id`` at ``t``."""
        node_id = self.net.edges[edge_id].v
        if not self._signalized[node_id]:
            return True
        return self.controller.allows(
            node_id, self._orientation[edge_id], TurnType.STRAIGHT, t)
