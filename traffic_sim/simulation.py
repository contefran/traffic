"""Step-based traffic simulation.

The whole dynamics live in :meth:`TrafficSim.step`: cars are grouped per edge,
sorted front-to-back, advanced with the Intelligent Driver Model (IDM), and
handed to the router when they reach the end of an edge. A red traffic light is
modelled as a stationary virtual obstacle at the stop line, so the same model
brakes for it. Density-dependent speed (the fundamental diagram) emerges from
IDM rather than being imposed.
"""

import math
from typing import Dict, List, Optional, Tuple

from .network import RoadNetwork
from .vehicles import Car
from .routing import RandomRouter
from .signals import SignalSystem, SignalState
from .priority import PriorityModel

# Hard safety buffer kept between a follower and its leader [m].
LEADER_BUFFER = 0.1
# IDM free-acceleration exponent (standard value).
IDM_DELTA = 4.0


class TrafficSim:
    """The mutable simulation state and its single-step update.

    Holds the (static) :class:`RoadNetwork`, the list of :class:`Car` objects it
    advances in place, and the pluggable policies that shape their behaviour: a
    ``router`` (which edge next), optional ``signals`` (traffic lights), optional
    ``priority`` (right-of-way at unsignalized nodes), and an optional
    ``metrics`` collector. Given a seeded router the whole thing is
    deterministic. Call :meth:`step` repeatedly to run it.
    """

    def __init__(
        self,
        net: RoadNetwork,
        cars: List[Car],
        router: Optional[RandomRouter] = None,
        signals: Optional[SignalSystem] = None,
        priority: Optional[PriorityModel] = None,
        metrics=None,
    ) -> None:
        """Wire up the simulation.

        ``router`` defaults to a :class:`RandomRouter` over ``net``. ``signals``
        of ``None`` means every approach is always green; ``priority`` of
        ``None`` means unsignalized nodes are an unchecked free-for-all;
        ``metrics`` of ``None`` means nothing is recorded. ``cars`` is mutated in
        place as the simulation runs.
        """
        self.net = net
        self.cars = cars
        self.router = router if router is not None else RandomRouter(net)
        self.signals = signals  # None => no signals, every approach is green
        # Right-of-way at unsignalized nodes; None => no yielding (free-for-all).
        self.priority = priority
        self.metrics = metrics  # optional MetricsCollector; observes each step
        self.t = 0.0
        # Genuine collisions: a car that could not stop in time even at its
        # physical braking limit (would have overlapped its leader or crossed a
        # red line). See the no-overlap constraint in :meth:`step`.
        self.crashes = 0

    def _unsignalized(self, node_id: int) -> bool:
        """True if ``node_id`` has no active signal (so right-of-way applies)."""
        return self.signals is None or not self.signals.is_signalized(node_id)

    def _approach_fronts(self, cars_on_edge: Dict[int, List[Car]]) -> Dict[int, list]:
        """Per unsignalized node, the front car of each approach that is near
        enough to contest, as ``(from_edge, to_edge, gap, speed)``. Commits each
        such car's ``next_edge`` so its intended movement is known.
        """
        fronts: Dict[int, list] = {}
        for edge_id, lst in cars_on_edge.items():
            edge = self.net.edges[edge_id]
            if not self._unsignalized(edge.v):
                continue
            front = lst[0]  # lst is sorted front (high s) -> back
            gap = edge.length - front.s
            if gap > self.priority.trigger_dist:
                continue
            if front.next_edge is None:
                front.next_edge = self.router.next_edge(edge_id, front)
            fronts.setdefault(edge.v, []).append(
                (edge_id, front.next_edge, gap, front.v))
        return fronts

    def _idm_accel(self, car: Car, v_des: float,
                   obstacle: Optional[Tuple[float, float]]) -> float:
        """Intelligent Driver Model acceleration.

        ``obstacle`` is ``(gap, lead_speed)`` for the constraint ahead — a real
        leader or a red stop line (a stationary obstacle, ``lead_speed=0``) — or
        ``None`` for open road. The free term accelerates toward ``v_des``; the
        interaction term brakes for the obstacle. Reuses the car's existing
        parameters: ``accel`` (a), ``braking`` (b), ``s0``, ``time_headway`` (T).
        """
        free = 1.0 - (car.v / v_des) ** IDM_DELTA if v_des > 0 else 0.0
        if obstacle is None:
            return car.accel * free

        gap, lead_v = obstacle
        approach_rate = car.v - lead_v  # > 0 when closing on the obstacle
        s_star = car.s0 + max(
            0.0,
            car.v * car.time_headway
            + (car.v * approach_rate) / (2.0 * math.sqrt(car.accel * car.braking)),
        )
        gap = max(gap, 0.01)  # guard against division blow-up at zero gap
        return car.accel * (free - (s_star / gap) ** 2)

    def step(self, dt: float) -> None:
        """Advance the whole simulation by ``dt`` seconds.

        In one pass: bucket cars by edge and sort each edge front-to-back;
        for each car compute its IDM acceleration against the most restrictive
        obstacle (its leader and/or a red stop line from signals or a
        right-of-way yield), cap deceleration at the car's physical limit
        (``max_brake``), then integrate speed and position. A no-overlap
        constraint still keeps a car from passing its leader or crossing a red
        line; because braking is now physically bounded, hitting that constraint
        means the car could not stop in time — a genuine collision, counted in
        ``self.crashes`` (the position is still clamped, so cars never visually
        overlap). Cars that reach the end of their edge are transferred to
        ``next_edge`` in a deferred second pass (carrying the overshoot) so a
        moving car does not disturb the ordering mid-step. Advances ``self.t``
        and, if a metrics collector is attached, records the new state.
        """
        # Group cars by edge and sort each edge front (high s) -> back.
        cars_on_edge: Dict[int, List[Car]] = {}
        for car in self.cars:
            cars_on_edge.setdefault(car.edge_id, []).append(car)
        for lst in cars_on_edge.values():
            lst.sort(key=lambda c: c.s, reverse=True)

        # Right-of-way contest data at unsignalized nodes (empty if disabled).
        fronts = self._approach_fronts(cars_on_edge) if self.priority is not None else {}

        # Defer edge transfers so a car moving to a new edge does not disturb
        # the leader/follower ordering of the edge currently being processed.
        transfers: List[tuple] = []  # (car, next_edge_id, new_s)

        for edge_id, lst in cars_on_edge.items():
            edge = self.net.edges[edge_id]

            for idx, car in enumerate(lst):
                leader = lst[idx - 1] if idx > 0 else None
                v_des = min(edge.speed_limit, car.max_speed)

                # Commit the next edge in advance so the signal can gate this
                # car's specific movement (e.g. a protected left vs a through).
                if car.next_edge is None:
                    car.next_edge = self.router.next_edge(edge_id, car)

                red = False
                if self.signals is not None and car.next_edge is not None:
                    state = self.signals.movement_state(edge_id, car.next_edge, self.t)
                    if state is SignalState.RED:
                        red = True
                    elif state is SignalState.YELLOW:
                        # Clearance: stop unless the car physically cannot — i.e.
                        # it could not halt before the line even at maximum
                        # braking. Only then is it committed and proceeds (clears
                        # on yellow instead of crashing the line). Using the
                        # physical limit, rather than comfortable braking, keeps
                        # cars from needlessly running the yellow into a
                        # downstream queue.
                        stop_dist = car.v * car.v / (2.0 * car.max_brake)
                        if edge.length - car.s >= stop_dist:
                            red = True

                # At an unsignalized node the front car of an approach may have
                # to yield right-of-way to conflicting higher-priority traffic.
                if (self.priority is not None and idx == 0
                        and self._unsignalized(edge.v)
                        and self.priority.must_yield(edge_id, car.next_edge,
                                                     fronts.get(edge.v, []))):
                    red = True

                # Constraints ahead, each (gap, speed): the leader and/or, on
                # red, the stop line at the end of the edge (a stopped object).
                obstacles = []
                if leader is not None:
                    obstacles.append((leader.s - car.s - car.length, leader.v))
                if red:
                    obstacles.append((edge.length - car.s, 0.0))

                # Most restrictive (smallest) IDM acceleration over obstacles.
                if obstacles:
                    a = min(self._idm_accel(car, v_des, o) for o in obstacles)
                else:
                    a = self._idm_accel(car, v_des, None)

                # Cap deceleration at the physical limit: a car cannot brake
                # harder than its tyres grip. (Acceleration is bounded by the IDM
                # free term already.) This is what makes an unavoidable overlap
                # below a *genuine* collision rather than a teleport-stop artifact.
                a = max(a, -car.max_brake)

                car.v = max(0.0, min(v_des, car.v + a * dt))
                new_s = car.s + car.v * dt

                # No-overlap constraint: a car may never pass its leader or cross
                # a red stop line. With deceleration now physically bounded,
                # reaching this constraint means even maximum braking was not
                # enough to stop in time — a real crash. We still clamp the
                # position (cars never visually overlap) but count the collision
                # and carry the leader's speed through (a rear-end, not a
                # teleport to zero).
                max_s = None
                if leader is not None:
                    max_s = leader.s - car.length - LEADER_BUFFER
                if red:
                    max_s = edge.length if max_s is None else min(max_s, edge.length)
                if max_s is not None and new_s > max_s:
                    self.crashes += 1
                    new_s = max(car.s, max_s)
                    car.v = min(car.v, leader.v if leader is not None else 0.0)

                # Reached the end of the edge.
                if new_s >= edge.length:
                    if car.next_edge is None:
                        new_s = edge.length  # dead-end: clamp and stop
                        car.v = 0.0
                    elif not red:
                        overshoot = new_s - edge.length
                        next_len = self.net.edges[car.next_edge].length
                        transfers.append((car, car.next_edge, min(overshoot, next_len)))
                        continue  # applied in the transfer pass
                    # If red, fall through: the car waits at the stop line.

                car.s = new_s
                car.trail.append((self.t, car.edge_id, car.s))

        for car, next_edge, new_s in transfers:
            car.edge_id = next_edge
            car.next_edge = None  # re-route from the new edge next step
            car.s = new_s
            car.trail.append((self.t, car.edge_id, car.s))

        self.t += dt

        if self.metrics is not None:
            self.metrics.record(self)
