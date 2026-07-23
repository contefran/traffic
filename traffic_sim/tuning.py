"""Flat parameter vector <-> live simulation knobs: the ML action surface.

Every optimizer we plan to point at the simulator (grid sweep, evolutionary
search, an RL policy) speaks the same language: a flat numeric vector. The
simulator's knobs, however, live in structured places — per-node
:class:`~traffic_sim.signals.SignalPlan` objects inside a controller, edge
``speed_limit`` attributes on the network. :class:`ParameterSpace` is the
adapter between the two: it enumerates the tunable dimensions once
(deterministic order), then ``vector()`` reads the current values and
``apply()`` writes a proposal back onto the live objects.

Two families of knobs are covered, matching the training-parameter priorities:

* **Signal timing** (per signalized node): the phase green times (cycle +
  split) and, optionally, the offset (green waves). Yellows are *not* in the
  vector — they follow the speed-scaled rule (see
  :func:`~traffic_sim.signals.apply_speed_scaled_yellows`), a safety
  constraint rather than an objective.
* **Speed limits** (per named edge group, e.g. ``{"local": [...], "arterial":
  [...]}``): one multiplier per group, scaling every member edge from its
  speed limit at construction time — so applying twice never compounds
  (same convention as the dashboard sliders).

``apply()`` **clips** each value into ``bounds()`` instead of raising, so a
raw optimizer proposal (a negative green, a wild multiplier) degrades to the
nearest legal plan rather than crashing a training run. If a ``router`` is
given, its cached cost tables are invalidated whenever speed multipliers are
applied, so cars re-route on the new speeds.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .network import RoadNetwork
from .signals import SignalPlan, SignalSystem, apply_speed_scaled_yellows


class ParameterSpace:
    """Maps a flat vector onto per-node signal plans and group speed limits.

    Layout (in :meth:`labels` order): for each signalized node, its
    ``NUM_PHASES`` green times followed by its offset (if ``include_offsets``);
    then one speed multiplier per group, groups in sorted-name order. Nodes are
    sorted by id, so the layout is deterministic for a given network.
    """

    def __init__(self, signals: Optional[SignalSystem] = None, *,
                 include_offsets: bool = True,
                 green_bounds: Tuple[float, float] = (3.0, 60.0),
                 offset_bound: Optional[float] = None,
                 speed_groups: Optional[Dict[str, Sequence[int]]] = None,
                 net: Optional[RoadNetwork] = None,
                 router=None,
                 speed_bounds: Tuple[float, float] = (0.5, 1.5)) -> None:
        """Enumerate the tunable dimensions of ``signals`` and ``speed_groups``.

        ``signals`` contributes the per-node timing dimensions (its controller
        must be plan-based, i.e. a :class:`PhasedController`); ``None`` gives a
        speeds-only space. ``speed_groups`` maps a group name to the edge ids
        it scales together (``net`` defaults to ``signals.net``); baselines are
        captured here, at construction. ``offset_bound`` caps the offset
        dimension (default: the longest possible cycle,
        ``NUM_PHASES * green_bounds[1]`` — offsets are periodic anyway).
        ``router`` is an optional :class:`ShortestPathRouter` whose cost cache
        is invalidated when speed multipliers are applied.
        """
        if signals is None and not speed_groups:
            raise ValueError("need signals and/or speed_groups to tune")
        self.signals = signals
        self.include_offsets = include_offsets
        self.green_bounds = green_bounds
        self.speed_bounds = speed_bounds
        self.router = router

        self.nodes: List[int] = []
        self.num_phases = 0
        if signals is not None:
            controller = signals.controller
            if not hasattr(controller, "plan_for"):
                raise TypeError("controller has no per-node plans to tune")
            self.controller = controller
            self.num_phases = len(controller.default_plan.green_times)
            self.nodes = sorted(n.id for n in signals.net.nodes
                                if signals.is_signalized(n.id))
        self.offset_bound = (offset_bound if offset_bound is not None
                             else self.num_phases * green_bounds[1])

        self.net = net or (signals.net if signals is not None else None)
        if speed_groups and self.net is None:
            raise ValueError("speed_groups need a net (or signals) to act on")
        self.group_names: List[str] = sorted(speed_groups) if speed_groups else []
        self._group_edges = {name: list(speed_groups[name])
                             for name in self.group_names}
        if any(not edges for edges in self._group_edges.values()):
            raise ValueError("speed groups must not be empty")
        # Baselines captured once so multipliers never compound.
        self._base_speed = {name: {eid: self.net.edges[eid].speed_limit
                                   for eid in edges}
                            for name, edges in self._group_edges.items()}

    # ------------------------------------------------------------------ shape

    @property
    def _per_node(self) -> int:
        """Vector dimensions contributed by each signalized node."""
        return self.num_phases + (1 if self.include_offsets else 0)

    @property
    def dim(self) -> int:
        """Total length of the parameter vector."""
        return len(self.nodes) * self._per_node + len(self.group_names)

    def labels(self) -> List[str]:
        """One human-readable name per dimension, in vector order."""
        out = []
        for node in self.nodes:
            out += [f"n{node}.green{k}" for k in range(self.num_phases)]
            if self.include_offsets:
                out.append(f"n{node}.offset")
        out += [f"speed.{name}" for name in self.group_names]
        return out

    def bounds(self) -> List[Tuple[float, float]]:
        """Per-dimension ``(lo, hi)`` box bounds, in vector order.

        ``apply`` clips into these, and an optimizer should sample within them.
        """
        per_node = [self.green_bounds] * self.num_phases
        if self.include_offsets:
            per_node.append((0.0, self.offset_bound))
        return per_node * len(self.nodes) + [self.speed_bounds] * len(self.group_names)

    # ------------------------------------------------------------- read/write

    def vector(self) -> List[float]:
        """The current parameter values, read from the live objects."""
        out: List[float] = []
        for node in self.nodes:
            plan = self.controller.plan_for(node)
            out += list(plan.green_times)
            if self.include_offsets:
                out.append(plan.offset)
        for name in self.group_names:
            base = self._base_speed[name]
            eid = self._group_edges[name][0]
            out.append(self.net.edges[eid].speed_limit / base[eid])
        return out

    def apply(self, vec: Sequence[float]) -> None:
        """Write ``vec`` onto the live controller / network (clipped to bounds).

        Each node gets a fresh :class:`SignalPlan` with the proposed green
        times and offset; its yellow is carried over from the plan it already
        had (yellows are a constraint, not a tunable — see module docstring).
        Speed multipliers rescale each group edge from its construction-time
        baseline, then the router cache (if any) is invalidated and, when the
        signal system carries speed-scaled yellows (``yellow_braking``), the
        scaling is re-applied so clearance times track the new approach speeds.
        (Note the scaling only ever *lengthens* a yellow above the floor: a
        node keeps its old, longer yellow if speeds are later lowered — safe,
        merely conservative.)
        """
        if len(vec) != self.dim:
            raise ValueError(f"expected {self.dim} values, got {len(vec)}")
        vec = [min(max(v, lo), hi) for v, (lo, hi) in zip(vec, self.bounds())]

        i = 0
        for node in self.nodes:
            greens = tuple(vec[i:i + self.num_phases])
            i += self.num_phases
            offset = vec[i] if self.include_offsets else \
                self.controller.plan_for(node).offset
            i += 1 if self.include_offsets else 0
            old = self.controller.plan_for(node)
            self.controller.set_plan(node, SignalPlan(
                greens, offset=offset, yellow=old.yellow))

        speeds_changed = False
        for name in self.group_names:
            mult = vec[i]
            i += 1
            base = self._base_speed[name]
            for eid in self._group_edges[name]:
                self.net.edges[eid].speed_limit = base[eid] * mult
            speeds_changed = True

        if speeds_changed:
            if self.router is not None:
                self.router.invalidate_costs()
            basis = getattr(self.signals, "yellow_braking", 0.0) \
                if self.signals is not None else 0.0
            if basis > 0:
                apply_speed_scaled_yellows(self.signals, basis)
