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
from .signals import SignalSystem, SignalState, TurnType, turn_type
from .priority import PriorityModel

# Hard safety buffer kept between a follower and its leader [m].
LEADER_BUFFER = 0.1
# IDM free-acceleration exponent (standard value).
IDM_DELTA = 4.0

# Cars slow to this speed to negotiate a turn (left / right / U) at an
# intersection; they brake for it comfortably over the approach so they reach the
# junction at ~this speed, then accelerate away on the exit street. Straight-
# through movements are unaffected. ~20 km/h.
TURN_SPEED = 20.0 / 3.6         # [m/s]

# Lane-change model (MOBIL-style): the acceleration gain a change must beat to be
# worth it, the hardest braking it may impose on the target lane's follower, and
# a small bonus that biases cars back toward the right (lower-index) lane.
LANE_CHANGE_MIN_GAIN = 0.2      # [m/s^2]
LANE_CHANGE_SAFE_BRAKE = 4.0    # [m/s^2]
KEEP_RIGHT_BONUS = 0.3          # [m/s^2]


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
        left_turn=None,
        parking=None,
        metrics=None,
        schedule=None,
    ) -> None:
        """Wire up the simulation.

        ``router`` defaults to a :class:`RandomRouter` over ``net``. ``signals``
        of ``None`` means every approach is always green; ``priority`` of
        ``None`` means unsignalized nodes are an unchecked free-for-all;
        ``left_turn`` of ``None`` means permissive lefts turn freely (no yielding
        to oncoming); ``metrics`` of ``None`` means nothing is recorded.
        ``schedule`` (a :class:`~traffic_sim.schedule.DailySchedule`) makes a car
        that parks *at its own home* sleep until its next morning departure;
        ``None`` uses the plain park-and-dwell for every stop. ``cars`` is mutated
        in place as the simulation runs.
        """
        self.net = net
        self.cars = cars
        self.router = router if router is not None else RandomRouter(net)
        self.signals = signals  # None => no signals, every approach is green
        # Right-of-way at unsignalized nodes; None => no yielding (free-for-all).
        self.priority = priority
        # Permissive-left gap acceptance at signalized nodes; None => free lefts.
        self.left_turn = left_turn
        # Park-and-dwell lifecycle; None => cars sail through destinations.
        self.parking = parking
        # Per-car daily routine; None => no home-overnight sleep (plain dwell).
        self.schedule = schedule
        self.metrics = metrics  # optional MetricsCollector; observes each step
        self.t = 0.0
        # Genuine collisions: a car that could not stop in time even at its
        # physical braking limit (would have overlapped its leader or crossed a
        # red line). See the no-overlap constraint in :meth:`step`.
        self.crashes = 0
        # Only run the lane-change pass if some edge actually has >1 lane.
        self._has_multilane = any(e.lanes > 1 for e in net.edges)
        # Cache of whether a movement is a turn (static per edge pair), so the
        # turn-slowdown doesn't recompute headings every step.
        self._is_turn_cache: Dict[Tuple[int, int], bool] = {}

    def _is_turn(self, from_edge: int, to_edge: int) -> bool:
        """Whether the ``from_edge -> to_edge`` movement is a turn (not straight).

        Left / right / U count; the result is cached (headings are static).
        """
        key = (from_edge, to_edge)
        cached = self._is_turn_cache.get(key)
        if cached is None:
            cached = turn_type(self.net, from_edge, to_edge) is not TurnType.STRAIGHT
            self._is_turn_cache[key] = cached
        return cached

    def _unsignalized(self, node_id: int) -> bool:
        """True if ``node_id`` has no active signal (so right-of-way applies)."""
        return self.signals is None or not self.signals.is_signalized(node_id)

    @staticmethod
    def _can_stop_within(car: Car, dist: float) -> bool:
        """Whether ``car`` can halt within ``dist`` metres at its physical brake
        limit — i.e. it is *before* its point of no return for a stop line ahead.

        Shared by every stop-line decision: the signal yellow clearance and the
        gap-acceptance yields (priority, permissive left). A car past this point
        cannot stop no matter what, so *forcing* it to yield would only
        manufacture a phantom collision at the line; instead it commits through,
        exactly as a car clears a yellow. Rear-ends behind a real leader are
        unaffected (a leader is a hard obstacle, not a stop-line decision)."""
        return dist >= car.v * car.v / (2.0 * car.max_brake)

    @staticmethod
    def _can_unpark(car: Car, lane_cars: List[Car]) -> bool:
        """Whether a parked ``car`` can pull out into its lane right now.

        Ahead, it needs a plain standstill gap to its would-be leader. Behind,
        it needs a **speed-aware** gap: an approaching follower must be far
        enough to brake for the emerging car within its own *comfortable*
        deceleration (``o.v²/2·braking`` plus a standstill margin ``s0``) — a
        real driver waits for a gap sized to oncoming *speed* before leaving a
        parking spot, not just a static car length. A purely static gap let a
        car pull out ~10 m in front of traffic doing 50 km/h, which then could
        not stop in time — the dominant residual-crash mechanism. Cautious /
        faster followers therefore demand a bigger gap; the parked car simply
        retries next step, pulling out the moment a safe gap opens.
        """
        for o in lane_cars:
            if o.s >= car.s:                        # would be car's leader ahead
                if o.s - car.s < car.length + car.s0:
                    return False
            else:                                   # approaching follower behind
                gap = car.s - o.s - o.length
                needed = o.s0 + o.v * o.v / (2.0 * o.braking)
                if gap < needed:
                    return False
        return True

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

    @staticmethod
    def _lane_neighbours(car: Car, lst: List[Car]):
        """Nearest ``(leader ahead, follower behind)`` of ``car`` in a lane list
        sorted front-first (high ``s`` -> low). ``car`` itself is skipped."""
        leader = follower = None
        for other in lst:
            if other is car:
                continue
            if other.s > car.s:
                leader = other          # nearest ahead = last one still ahead
            else:
                follower = other        # first one behind
                break
        return leader, follower

    @staticmethod
    def _gap_obstacle(s: float, length: float, lead: Optional[Car]):
        """`(gap, lead_speed)` from a car at ``s`` to ``lead``, or ``None``."""
        return None if lead is None else (lead.s - s - length, lead.v)

    def _apply_lane_changes(self, cars_on_lane: Dict[Tuple[int, int], List[Car]]) -> None:
        """Apply MOBIL-style lane changes, **in place, one car at a time**.

        A car moves to an adjacent lane when it gains acceleration there
        (``a_new - a_old`` beats :data:`LANE_CHANGE_MIN_GAIN`, plus a keep-right
        bonus) *and* it is safe — it fits ahead of that lane's leader and does not
        force that lane's follower to brake harder than
        :data:`LANE_CHANGE_SAFE_BRAKE`. Each move updates ``cars_on_lane``
        immediately, so a later car sees it and two cars never merge into the same
        gap on the same step (which would otherwise collide in dense traffic).
        """
        # Snapshot the (edge, lane) keys and their cars up front; each car is
        # considered once, but neighbour lookups read the live, updated buckets.
        for (edge_id, lane) in [k for k in cars_on_lane if self.net.edges[k[0]].lanes > 1]:
            edge = self.net.edges[edge_id]
            for car in list(cars_on_lane.get((edge_id, lane), ())):
                v_des = min(edge.speed_limit, car.max_speed)
                cur_leader, _ = self._lane_neighbours(car, cars_on_lane.get((edge_id, car.lane), ()))
                a_old = self._idm_accel(car, v_des,
                                        self._gap_obstacle(car.s, car.length, cur_leader))
                best_lane, best_gain = car.lane, LANE_CHANGE_MIN_GAIN
                for target in (car.lane - 1, car.lane + 1):
                    if not 0 <= target < edge.lanes:
                        continue
                    leader, follower = self._lane_neighbours(
                        car, cars_on_lane.get((edge_id, target), ()))
                    if leader is not None and leader.s - car.s - car.length < LEADER_BUFFER:
                        continue                                     # would overlap ahead
                    if follower is not None:
                        gap = car.s - follower.s - follower.length
                        if gap < LEADER_BUFFER:
                            continue                                 # would overlap behind
                        fv_des = min(edge.speed_limit, follower.max_speed)
                        if self._idm_accel(follower, fv_des, (gap, car.v)) < -LANE_CHANGE_SAFE_BRAKE:
                            continue                                 # cuts the follower off
                    a_new = self._idm_accel(car, v_des,
                                            self._gap_obstacle(car.s, car.length, leader))
                    # Safety is symmetric (standard MOBIL): the *mover* must also
                    # be able to follow its new leader without braking harder than
                    # LANE_CHANGE_SAFE_BRAKE. Without this a car already braking in
                    # its own lane (very negative a_old) could still net a positive
                    # gain by diving into a tiny gap behind a slow leader — a cut-in
                    # it then rear-ends next step. The geometric overlap check above
                    # only guards the instant, not the stopping distance.
                    if a_new < -LANE_CHANGE_SAFE_BRAKE:
                        continue
                    gain = a_new - a_old + (KEEP_RIGHT_BONUS if target < car.lane else 0.0)
                    if gain > best_gain:
                        best_gain, best_lane = gain, target
                if best_lane != car.lane:
                    cars_on_lane[(edge_id, car.lane)].remove(car)   # leave old lane
                    car.lane = best_lane
                    tl = cars_on_lane.setdefault((edge_id, best_lane), [])
                    tl.append(car)
                    tl.sort(key=lambda c: c.s, reverse=True)         # keep front-first

    def step(self, dt: float) -> None:
        """Advance the whole simulation by ``dt`` seconds.

        In one pass: bucket cars by ``(edge, lane)`` and sort each lane
        front-to-back (a car follows only the leader in its own lane); also keep
        by-edge buckets for the intersection logic. For each car compute its IDM
        acceleration against the most restrictive
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
        # Keep the router's clock in sync so a time-of-day demand model can pick
        # destinations for the current moment (a no-op for time-agnostic routers).
        self.router.now = self.t

        # Wake pass: parked cars whose dwell is over re-enter the flow and head to
        # their next destination — the next leg of their activity plan if a
        # schedule owns it, otherwise a fresh one from the router. A car only
        # pulls out into a real **gap** at its kerbside spot (leader and follower
        # clearance in its lane); if traffic is queued over the spot it stays
        # parked and retries next step — re-entering regardless would inject an
        # overlapping car, a placement artifact the crash counter would then
        # score every step (not a genuine collision).
        if self.parking is not None:
            waking = [c for c in self.cars if not c.active and self.t >= c.wake_t]
            if waking:
                lane_cars: Dict[Tuple[int, int], List[Car]] = {}
                for c in self.cars:
                    if c.active:
                        lane_cars.setdefault((c.edge_id, c.lane), []).append(c)
                for car in waking:
                    slot = lane_cars.setdefault((car.edge_id, car.lane), [])
                    if not self._can_unpark(car, slot):
                        continue                     # blocked in; try next step
                    car.active = True
                    car.next_edge = None
                    slot.append(car)                 # claims the spot this step
                    if self.schedule is not None and hasattr(self.schedule, "on_wake"):
                        self.schedule.on_wake(car, self.t)
                    elif hasattr(self.router, "assign_destination"):
                        self.router.assign_destination(
                            car, avoid=self.net.edges[car.edge_id].v)

        # Group *active* cars by edge (all lanes) for intersection logic, and by
        # (edge, lane) for car-following. Each list is sorted front (high s) -> back.
        cars_on_edge: Dict[int, List[Car]] = {}
        cars_on_lane: Dict[Tuple[int, int], List[Car]] = {}
        for car in self.cars:
            if not car.active:
                continue  # parked cars are off the road
            cars_on_edge.setdefault(car.edge_id, []).append(car)
            cars_on_lane.setdefault((car.edge_id, car.lane), []).append(car)
        for lst in cars_on_edge.values():
            lst.sort(key=lambda c: c.s, reverse=True)
        for lst in cars_on_lane.values():
            lst.sort(key=lambda c: c.s, reverse=True)

        # Lane-change pass (overtaking): applied car-by-car in place, so two cars
        # never move into the same gap on the same step (which would collide).
        if self._has_multilane:
            self._apply_lane_changes(cars_on_lane)

        # Right-of-way contest data at unsignalized nodes (empty if disabled).
        fronts = self._approach_fronts(cars_on_edge) if self.priority is not None else {}

        # Defer edge transfers so a car moving to a new edge does not disturb
        # the leader/follower ordering of the edge currently being processed.
        transfers: List[tuple] = []  # (car, next_edge_id, new_s, new_lane)

        for (edge_id, _lane), lst in cars_on_lane.items():
            edge = self.net.edges[edge_id]

            for idx, car in enumerate(lst):
                leader = lst[idx - 1] if idx > 0 else None  # nearest in same lane
                v_des = min(edge.speed_limit, car.max_speed)

                # Commit the next edge in advance so the signal can gate this
                # car's specific movement (e.g. a protected left vs a through).
                # With parking on, a car that has reached its destination edge
                # does not route onward — it keeps ``next_edge = None`` so it
                # stops at the node and is parked in the pass below.
                # Arrived when on the specific destination street (edge-precise),
                # or — for a manually set node destination without a dest_edge —
                # on any edge into the destination node (the old behaviour).
                arriving = (self.parking is not None and (
                    (car.dest_edge is not None and edge_id == car.dest_edge)
                    or (car.dest_edge is None and car.dest is not None
                        and edge.v == car.dest)))
                # The destination point along this final edge (mid-block if
                # ``dest_frac`` < 1, else the node); the car stops here.
                stop_pos = edge.length * car.dest_frac if arriving else None
                if arriving:
                    # An arriving car does not route onward; clear any stale
                    # commitment so the park pass (which skips cars with a
                    # committed next_edge) can actually park it — otherwise a car
                    # that reached its street with a left-over next_edge never
                    # parks and deadlocks the lane behind it.
                    car.next_edge = None
                elif car.next_edge is None:
                    car.next_edge = self.router.next_edge(edge_id, car)

                # Slow down for a turn ahead: cap the desired speed so the car can
                # brake comfortably to ~TURN_SPEED by the intersection, then take
                # the corner and speed up again on the exit street. The cap is the
                # kinematic "safe speed" v = sqrt(v_turn^2 + 2*b*dist), which is
                # high far out (no effect) and falls to TURN_SPEED at the junction.
                if car.next_edge is not None and self._is_turn(edge_id, car.next_edge):
                    dist = edge.length - car.s
                    v_turn_cap = math.sqrt(
                        TURN_SPEED * TURN_SPEED + 2.0 * car.braking * max(0.0, dist))
                    v_des = min(v_des, v_turn_cap)

                red = False
                sig_state = None
                if self.signals is not None and car.next_edge is not None:
                    sig_state = self.signals.movement_state(edge_id, car.next_edge, self.t)
                    if sig_state is SignalState.RED:
                        red = True
                    elif sig_state is SignalState.YELLOW:
                        # Clearance: stop unless the car is past its point of no
                        # return, in which case it commits and clears the yellow
                        # instead of crashing the line (see _can_stop_within).
                        if self._can_stop_within(car, edge.length - car.s):
                            red = True

                # Gap-acceptance yields below only bind a car that can *still*
                # stop for the line: a car past its point of no return is already
                # committed into the junction, so forcing a yield it cannot honour
                # would only manufacture a phantom crash at the stop line (this was
                # the whole `red`-cause crash family — all at unsignalized/priority
                # yields). A driver accepts or rejects the gap *before* committing;
                # this makes the model do the same, exactly as the yellow does.
                can_still_stop = self._can_stop_within(car, edge.length - car.s)

                # Permissive left: a front left-turner on a *full* green must
                # yield to imminent oncoming through traffic (inert under
                # protected phasing, where that traffic is red). Only at
                # *signalized* nodes — at unsignalized ones (e.g. elevated ramp
                # merges) movement_state is GREEN by default and right-of-way is
                # the PriorityModel's job; applying it there would spuriously yield
                # forever and deadlock the merge.
                if (self.left_turn is not None and idx == 0
                        and sig_state is SignalState.GREEN and can_still_stop
                        and not self._unsignalized(edge.v)
                        and car.next_edge is not None
                        and self.left_turn.must_yield(edge_id, car.next_edge,
                                                      self.signals, self.t, cars_on_edge)):
                    red = True

                # At an unsignalized node the front car of an approach may have
                # to yield right-of-way to conflicting higher-priority traffic.
                if (self.priority is not None and idx == 0 and can_still_stop
                        and self._unsignalized(edge.v)
                        and self.priority.must_yield(edge_id, car.next_edge,
                                                     fronts.get(edge.v, []))):
                    red = True

                # Constraints ahead, each (gap, speed): the leader, a red stop
                # line at the end of the edge, and/or (when arriving) the
                # destination point mid-block — all stationary obstacles.
                obstacles = []
                if leader is not None:
                    obstacles.append((leader.s - car.s - car.length, leader.v))
                if red:
                    obstacles.append((edge.length - car.s, 0.0))
                if stop_pos is not None:
                    obstacles.append((stop_pos - car.s, 0.0))
                # Spillback look-ahead: the front car also brakes for the queue
                # tail on its committed next edge — just a leader seen across
                # the junction. Without this, a car is blind past the node and
                # sails into a street that is jammed back to its start.
                if idx == 0 and car.next_edge is not None:
                    nxt = self.net.edges[car.next_edge]
                    slot = cars_on_lane.get(
                        (car.next_edge, min(car.lane, nxt.lanes - 1)))
                    if slot:
                        tail = slot[-1]         # deepest car on the next edge
                        obstacles.append(
                            ((edge.length - car.s) + tail.s - car.length, tail.v))

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

                # No-overlap constraint: a car may never pass the nearest stop
                # ahead. That stop is a **hard obstacle** — its leader or a red
                # stop line — or the car's **own destination point** (the address
                # it is parking at). With deceleration physically bounded,
                # crossing a *hard* obstacle despite maximum braking is a genuine
                # collision (counted, leader's speed carried through — a rear-end,
                # not a teleport to zero). Crossing one's own destination point is
                # an **arrival**, not a crash: it clamps and comes to rest without
                # counting (a car rolling the last centimetres into its parking
                # spot is not a collision — this was ~2/3 of the old crash count).
                # The binding (nearest) constraint decides which case applies.
                max_s, hard = None, False
                if leader is not None:
                    max_s, hard = leader.s - car.length - LEADER_BUFFER, True
                if red and (max_s is None or edge.length < max_s):
                    max_s, hard = edge.length, True
                if stop_pos is not None and (max_s is None or stop_pos < max_s):
                    max_s, hard = stop_pos, False     # own address is the binding stop
                if max_s is not None and new_s > max_s:
                    if hard:
                        # A collision needs closing speed. If the obstacle ahead
                        # is moving at least as fast as the car (a leader pulling
                        # away, a merge in behind a faster car), any overlap is a
                        # one-step position artifact — not an impact — so clamp it
                        # out but do not count it. A stationary red line (lead_v 0)
                        # with the car still moving is a genuine run of the line.
                        lead_v = leader.v if leader is not None else 0.0
                        if car.v > lead_v:
                            self.crashes += 1
                        car.v = min(car.v, lead_v)
                    else:
                        car.v = 0.0                    # arrived: come to rest here
                    new_s = max(car.s, max_s)

                # Reached the end of the edge.
                if new_s >= edge.length:
                    if car.next_edge is None:
                        new_s = edge.length  # dead-end: clamp and stop
                        car.v = 0.0
                    elif not red:
                        overshoot = new_s - edge.length
                        next_edge = self.net.edges[car.next_edge]
                        # Merge into a valid lane on the next edge (clamp index).
                        new_lane = min(car.lane, next_edge.lanes - 1)
                        transfers.append((car, car.next_edge,
                                          min(overshoot, next_edge.length), new_lane))
                        continue  # applied in the transfer pass
                    # If red, fall through: the car waits at the stop line.

                car.s = new_s
                car.trail.append((self.t, car.edge_id, car.s))

        for car, next_eid, new_s, new_lane in transfers:
            # A transfer never lands a car on top of the queue on its new edge:
            # it slots in behind the current tail. If the street is full back to
            # the junction, the car waits at the stop line instead of entering
            # the box (spillback, not an overlap); if it lands harder than the
            # tail allows despite the look-ahead, that is a genuine rear-end
            # while crossing — counted once, then resolved (position clamped,
            # tail's speed carried), so no overlap ever persists.
            slot = cars_on_lane.setdefault((next_eid, new_lane), [])
            tail = slot[-1] if slot else None
            if tail is not None:
                allowed = tail.s - car.length - LEADER_BUFFER
                if allowed < 0.0:                 # no room past the junction
                    car.s = self.net.edges[car.edge_id].length
                    car.v = 0.0                   # wait at the line, retry later
                    continue                      # (keeps next_edge committed)
                if new_s > allowed:
                    # Only a car closing on the tail rear-ends it; entering behind
                    # a faster tail that is pulling away is a one-step position
                    # artifact (clamp, don't count) — same closing-speed rule as
                    # the in-lane check above.
                    if car.v > tail.v:
                        self.crashes += 1        # rear-ended the queue tail
                    car.v = min(car.v, tail.v)
                    new_s = allowed
                # Entry-speed cap: crawl into a backed-up junction rather than
                # blast in at road speed. Cap the entry speed to what still lets
                # the car brake (comfortably) behind the tail it is merging in
                # behind — the merge analogue of the turn slowdown. Without it a
                # car could geometrically *fit* just behind a near-stopped tail
                # yet keep 10 m/s, guaranteeing a rear-end the very next step; the
                # look-ahead can miss this when the queue forms during the crossing
                # step. A clear entrance (tail far) leaves the cap slack, so free
                # junctions are unaffected.
                gap = tail.s - new_s - car.length
                v_cap = tail.v + math.sqrt(2.0 * car.braking * max(0.0, gap))
                car.v = min(car.v, v_cap)
            # Leave the old lane bucket before mutating the car, so a later
            # transfer targeting the old edge never sees a phantom tail whose
            # ``s`` is already in new-edge coordinates.
            old_slot = cars_on_lane.get((car.edge_id, car.lane))
            if old_slot is not None and car in old_slot:
                old_slot.remove(car)
            car.edge_id = next_eid
            car.next_edge = None  # re-route from the new edge next step
            car.s = new_s
            car.lane = new_lane
            car.trail.append((self.t, car.edge_id, car.s))
            slot.append(car)      # now the deepest car: visible to later transfers

        # Park pass: a car halted at its destination node goes inactive for a
        # dwell (the wake pass re-enters it later with a fresh destination).
        if self.parking is not None:
            for car in self.cars:
                if not car.active or car.dest is None or car.next_edge is not None:
                    continue
                edge = self.net.edges[car.edge_id]
                stop_pos = edge.length * car.dest_frac      # mid-block if <1
                at_dest = (car.edge_id == car.dest_edge if car.dest_edge is not None
                           else edge.v == car.dest)
                if (at_dest and car.v <= 0.5 and stop_pos - car.s <= 3.0):
                    car.active = False
                    if self.schedule is not None and hasattr(self.schedule, "on_park"):
                        # Sleep/dwell until this car's next scheduled activity.
                        self.schedule.on_park(car, self.t)
                    elif self.schedule is not None and hasattr(self.schedule, "next_departure") \
                            and car.edge_id == car.home:
                        # (legacy DailySchedule) home for the night.
                        car.wake_t = self.schedule.next_departure(car, self.t)
                    else:
                        # Dwell keyed on the street parked on (land use on edges).
                        car.wake_t = self.t + self.parking.dwell_time(car.edge_id)

        self.t += dt

        if self.metrics is not None:
            self.metrics.record(self)
