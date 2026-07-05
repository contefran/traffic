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
left turns their own protected phase). Both are **decentralized**: each
intersection carries its own :class:`SignalPlan` (cycle / split / offset), so
there is no global clock — nodes without an explicit plan share a default and
happen to run in phase. This is the surface an adaptive / learned controller
tunes later.

Nodes carrying only one orientation (grid borders) are never signalized; a node
may also be forced unsignalized via ``SignalSystem(unsignalized_nodes=...)``.
NOTE: an unsignalized node currently has **no right-of-way** modelling — all
movements are permitted, so conflicting traffic passes through unchecked. That
is a known, deliberate gap.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Protocol, Set, Tuple

from .network import RoadNetwork


class Orientation(Enum):
    """The axis an approach runs along, used to group movements into phases."""

    HORIZONTAL = "H"  # edge runs east-west (endpoints share a grid row j)
    VERTICAL = "V"    # edge runs north-south (endpoints share a grid column i)


class TurnType(Enum):
    """A movement's turn relative to its approach, from :func:`turn_type`."""

    STRAIGHT = "S"
    LEFT = "L"
    RIGHT = "R"
    UTURN = "U"


class SignalState(Enum):
    """Three-colour state of a movement's signal.

    ``YELLOW`` is the clearance interval after green: a car stops if it still
    can (comfortably), otherwise it is committed and proceeds through — which is
    how :meth:`TrafficSim.step` interprets it. Controllers built without a yellow
    interval only ever return ``GREEN`` / ``RED`` (backward compatible).
    """

    GREEN = "G"
    YELLOW = "Y"
    RED = "R"


def edge_orientation(net: RoadNetwork, edge_id: int) -> Orientation:
    """Classify an edge as horizontal or vertical from its endpoints' grid rows.

    Uses the grid indices (``u.j == v.j`` means both endpoints share a row, i.e.
    an east-west edge), so it is unaffected by positional jitter.
    """
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
        """Return whether the movement is permitted at ``node_id`` at time ``t``.

        Implementations receive the approach ``orientation`` and the ``turn``
        type and decide from the node's current phase. Implement this to plug in
        an adaptive or learned controller.
        """
        ...


@dataclass
class SignalPlan:
    """Per-intersection timing: the *decentralized* knobs, one plan per node.

    * ``green_times`` — green duration of each phase, in phase order. Its length
      is the number of phases; its **sum is the cycle length**; unequal values
      are the phase **split**.
    * ``offset`` — shifts this node's schedule relative to global time ``t``, so
      neighbouring intersections need not switch together (enabling green waves).
    * ``yellow`` — a clearance interval appended to *each* phase after its green,
      during which that phase's movements are :attr:`SignalState.YELLOW` (stop if
      you can, else commit). ``0`` (the default) reproduces the old two-colour
      behaviour. It lengthens the cycle by ``yellow`` per phase.

    A controller keeps one plan per node and a default; two nodes with different
    offsets run on independent clocks. These are exactly the parameters an
    adaptive / learned controller would tune later.
    """

    green_times: Tuple[float, ...]
    offset: float = 0.0
    yellow: float = 0.0

    def __post_init__(self) -> None:
        """Validate the phase durations."""
        if not self.green_times or any(g <= 0 for g in self.green_times):
            raise ValueError("green_times must be non-empty and positive")
        if self.yellow < 0:
            raise ValueError("yellow must be non-negative")

    @property
    def cycle(self) -> float:
        """Total cycle length [s]: the green times plus a yellow after each."""
        return sum(self.green_times) + len(self.green_times) * self.yellow

    def phase_state(self, t: float) -> Tuple[int, "SignalState"]:
        """The ``(phase index, GREEN or YELLOW)`` active at time ``t``.

        Each phase occupies ``green_times[k]`` of green followed by ``yellow`` of
        yellow; a movement's own phase is thus green then yellow, and red during
        every other phase (that comparison lives in the controller).
        """
        tau = (t - self.offset) % self.cycle
        acc = 0.0
        for k, g in enumerate(self.green_times):
            if tau < acc + g:
                return k, SignalState.GREEN
            if tau < acc + g + self.yellow:
                return k, SignalState.YELLOW
            acc += g + self.yellow
        return len(self.green_times) - 1, SignalState.YELLOW  # float seam

    def active_phase(self, t: float) -> int:
        """Index of the phase active at time ``t`` (green or its trailing yellow)."""
        return self.phase_state(t)[0]


class PhasedController:
    """Base for time-driven controllers: each node follows its own
    :class:`SignalPlan` (offset + split), so there is **no global clock**. Nodes
    without an explicit plan fall back to a shared default (all such nodes then
    run in phase — the original uniform behaviour). Subclasses set the number of
    phases and map a phase index to the movements it permits.
    """

    NUM_PHASES: int = 0

    def __init__(self, green_time: float, *, yellow: float = 0.0,
                 plans: Optional[Dict[int, SignalPlan]] = None,
                 default_plan: Optional[SignalPlan] = None) -> None:
        """Set up the shared default plan and any per-node overrides.

        ``green_time`` builds the default :class:`SignalPlan` (all
        :attr:`NUM_PHASES` phases equal, zero offset); ``yellow`` is its clearance
        interval per phase (``0`` = two-colour). ``default_plan`` replaces that
        fallback outright; ``plans`` is an optional initial ``{node_id:
        SignalPlan}`` mapping (more can be added later with :meth:`set_plan`).
        """
        self.default_plan = default_plan or SignalPlan(
            (green_time,) * self.NUM_PHASES, yellow=yellow)
        self.plans: Dict[int, SignalPlan] = dict(plans or {})

    def set_plan(self, node_id: int, plan: SignalPlan) -> None:
        """Give one intersection its own timing (offset / split / cycle / yellow)."""
        self.plans[node_id] = plan

    def plan_for(self, node_id: int) -> SignalPlan:
        """The node's own :class:`SignalPlan`, or the shared default."""
        return self.plans.get(node_id, self.default_plan)

    def phase(self, node_id: int, t: float) -> int:
        """Index of the phase active at ``node_id`` at time ``t``."""
        return self.plan_for(node_id).active_phase(t)

    def _movement_phase(self, orientation: Orientation, turn: TurnType) -> int:
        """The phase index that serves this movement. Subclasses implement it."""
        raise NotImplementedError

    def state(self, node_id: int, orientation: Orientation,
              turn: TurnType, t: float) -> SignalState:
        """Three-colour signal for a movement: its phase's green/yellow, else red.

        The movement is :attr:`SignalState.GREEN` or ``YELLOW`` exactly while its
        own phase (:meth:`_movement_phase`) is the active one, and ``RED``
        otherwise. With no yellow interval this only ever returns green or red.
        """
        active, sub = self.plan_for(node_id).phase_state(t)
        if self._movement_phase(orientation, turn) == active:
            return sub
        return SignalState.RED

    def allows(self, node_id: int, orientation: Orientation,
               turn: TurnType, t: float) -> bool:
        """Whether the movement has a full green now (``state`` is ``GREEN``)."""
        return self.state(node_id, orientation, turn, t) is SignalState.GREEN


class FixedTimeController(PhasedController):
    """Simple 2-phase light: each intersection alternates H/V on its own timer.

    Turns are *permissive* — any turn is allowed whenever your approach
    orientation has green, so a left turn can conflict with oncoming through
    traffic. Timing is per-node via :class:`SignalPlan` (see base class).
    """

    NUM_PHASES = 2

    def __init__(self, green_time: float = 10.0,
                 start: Orientation = Orientation.HORIZONTAL, *,
                 yellow: float = 0.0,
                 plans: Optional[Dict[int, SignalPlan]] = None,
                 default_plan: Optional[SignalPlan] = None) -> None:
        """Build a 2-phase fixed-time controller.

        ``green_time`` is the default green duration of each of the two phases and
        ``yellow`` their clearance interval. ``start`` is the orientation that is
        green in phase 0 (at ``t = offset``). ``plans`` optionally supplies
        per-node :class:`SignalPlan` objects up front; ``default_plan`` overrides
        the plan used by nodes that have none. See :class:`PhasedController` for
        how plans are resolved per node.
        """
        super().__init__(green_time, yellow=yellow, plans=plans,
                         default_plan=default_plan)
        self.start = start

    def green_orientation(self, node_id: int, t: float) -> Orientation:
        """The orientation that currently has green at ``node_id`` at time ``t``.

        Phase 0 gives green to :attr:`start`; phase 1 gives it to the other
        orientation. Which phase is active depends on the node's own plan
        (offset and split), so two nodes can show different green orientations at
        the same instant. (During a phase's yellow this still reports that
        phase's orientation.)
        """
        other = (Orientation.VERTICAL if self.start is Orientation.HORIZONTAL
                 else Orientation.HORIZONTAL)
        return other if self.phase(node_id, t) == 1 else self.start

    def _movement_phase(self, orientation: Orientation, turn: TurnType) -> int:
        """Phase 0 serves the ``start`` orientation, phase 1 the other.

        Turns are *permissive* — any turn goes with its approach orientation, so
        a left shares its phase with oncoming through traffic and may conflict.
        Use :class:`ProtectedPhaseController` if that conflict matters.
        """
        return 0 if orientation is self.start else 1


class ProtectedPhaseController(PhasedController):
    """4-phase cycle giving left turns a protected phase, timed per-node.

    Phase order (durations come from each node's :class:`SignalPlan`):
      0  H through + right
      1  H left (and U-turn)
      2  V through + right
      3  V left (and U-turn)

    Because through and left never share a phase, a protected left never crosses
    oncoming through traffic.
    """

    NUM_PHASES = 4

    _LABELS = ("H→", "H↰", "V→", "V↰")

    def __init__(self, green_time: float = 5.0, *, yellow: float = 0.0,
                 plans: Optional[Dict[int, SignalPlan]] = None,
                 default_plan: Optional[SignalPlan] = None) -> None:
        """Build a 4-phase protected-left controller.

        ``green_time`` is the default green duration of each of the four phases
        (H-through, H-left, V-through, V-left) and ``yellow`` their clearance
        interval; together they form the default :class:`SignalPlan`. ``plans``
        optionally supplies per-node plans up front and ``default_plan`` overrides
        the fallback plan; see :class:`PhasedController` for per-node resolution.
        """
        super().__init__(green_time, yellow=yellow, plans=plans,
                         default_plan=default_plan)

    def phase_label(self, node_id: int, t: float) -> str:
        """A short human-readable label for the active phase (e.g. ``"H→"``).

        Handy for annotating renders/animations. The four labels correspond to
        H-through, H-left, V-through, V-left in phase order.
        """
        return self._LABELS[self.phase(node_id, t)]

    def _movement_phase(self, orientation: Orientation, turn: TurnType) -> int:
        """Phase serving a movement under protected phasing.

        Through (and right) movements are served in phase 0 (horizontal) or 2
        (vertical); left turns (and U-turns) get their own phase, 1 (horizontal)
        or 3 (vertical). Because a left never shares a phase with the opposing
        through movement, protected lefts never cross oncoming traffic.
        """
        is_left = turn in (TurnType.LEFT, TurnType.UTURN)
        if orientation is Orientation.HORIZONTAL:
            return 1 if is_left else 0
        return 3 if is_left else 2


class SignalSystem:
    """Applies a :class:`SignalController` to a network and answers the sim.

    Decides which nodes actually carry signals, then translates a car's concrete
    edge-to-edge movement into the ``(orientation, turn)`` question the
    controller understands. A node is signalized only if it genuinely mixes both
    orientations (i.e. it is a real crossing, not a mid-street point) and has not
    been forced off via ``unsignalized_nodes``. Unsignalized nodes permit every
    movement — right-of-way there, if any, is handled separately by
    :class:`~traffic_sim.priority.PriorityModel`.
    """

    def __init__(self, net: RoadNetwork, controller: SignalController,
                 unsignalized_nodes: Optional[Set[int]] = None) -> None:
        """Precompute per-node signalization for ``net`` under ``controller``.

        ``controller`` supplies the timing/phasing policy. ``unsignalized_nodes``
        is an optional set of node ids to force off (kept unsignalized even if
        they mix orientations) — useful for experiments with uncontrolled
        junctions. Edge orientations are cached here since the network is static.
        """
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
        """Whether ``node_id`` carries an active traffic signal.

        ``False`` for grid-border nodes that only see one orientation and for any
        node listed in ``unsignalized_nodes``; such nodes allow all movements.
        """
        return self._signalized[node_id]

    def allows(self, node_id: int, orientation: Orientation,
               turn: TurnType, t: float) -> bool:
        """Node-level movement query, phrased in ``(orientation, turn)`` terms.

        Mainly used by the renderer to colour each movement indicator. Returns
        ``True`` unconditionally at unsignalized nodes; otherwise defers to the
        controller. See :meth:`allows_movement` for the edge-based form the
        simulation uses.
        """
        if not self._signalized[node_id]:
            return True
        return self.controller.allows(node_id, orientation, turn, t)

    def allows_movement(self, from_edge: int, to_edge: int, t: float) -> bool:
        """Whether a car has a full green to cross ``from_edge`` -> ``to_edge``.

        The movement is resolved to the approach orientation of ``from_edge`` and
        the :func:`turn_type` of the turn, then passed to the controller. Always
        ``True`` at an unsignalized node. See :meth:`movement_state` for the
        three-colour form the simulation uses (so it can act on yellow).
        """
        node_id = self.net.edges[from_edge].v
        if not self._signalized[node_id]:
            return True
        orientation = self._orientation[from_edge]
        turn = turn_type(self.net, from_edge, to_edge)
        return self.controller.allows(node_id, orientation, turn, t)

    def _controller_state(self, node_id: int, orientation: Orientation,
                          turn: TurnType, t: float) -> SignalState:
        """The controller's three-colour state, deriving it from ``allows`` for
        controllers that predate :meth:`SignalController.state`."""
        state_fn = getattr(self.controller, "state", None)
        if state_fn is not None:
            return state_fn(node_id, orientation, turn, t)
        return (SignalState.GREEN
                if self.controller.allows(node_id, orientation, turn, t)
                else SignalState.RED)

    def state(self, node_id: int, orientation: Orientation,
              turn: TurnType, t: float) -> SignalState:
        """Three-colour state for a movement (green at an unsignalized node).

        Used by the renderer to colour each movement indicator green / amber /
        red."""
        if not self._signalized[node_id]:
            return SignalState.GREEN
        return self._controller_state(node_id, orientation, turn, t)

    def movement_state(self, from_edge: int, to_edge: int, t: float) -> SignalState:
        """Three-colour signal for a car crossing ``from_edge`` -> ``to_edge``.

        This is what :meth:`TrafficSim.step` queries: ``GREEN`` proceeds, ``RED``
        stops, and ``YELLOW`` means stop only if the car can still do so
        comfortably (else it commits through). Always ``GREEN`` at an
        unsignalized node.
        """
        node_id = self.net.edges[from_edge].v
        if not self._signalized[node_id]:
            return SignalState.GREEN
        orientation = self._orientation[from_edge]
        turn = turn_type(self.net, from_edge, to_edge)
        return self._controller_state(node_id, orientation, turn, t)

    def is_green(self, edge_id: int, t: float) -> bool:
        """Convenience: may a car proceed *straight* off ``edge_id`` at ``t``?

        A shorthand for the common through-movement case (equivalent to
        :meth:`allows` with :attr:`TurnType.STRAIGHT` for this edge's
        orientation). Always ``True`` at an unsignalized node.
        """
        node_id = self.net.edges[edge_id].v
        if not self._signalized[node_id]:
            return True
        return self.controller.allows(
            node_id, self._orientation[edge_id], TurnType.STRAIGHT, t)
