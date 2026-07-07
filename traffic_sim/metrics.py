"""Diagnostics: record traffic-flow quantities as the simulation runs.

These are the fundamental variables of traffic flow — speed, density (here as
occupancy/queueing), and throughput — and the intended training signal for the
later flow-prediction model. The collector is dependency-free and observes the
simulation from the outside, so the simulation core stays unaware of it.

Inject one into ``TrafficSim`` (it calls ``record(sim)`` after each step) or
call ``record(sim)`` yourself in a custom loop.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# A car slower than this is treated as queued/stopped [m/s].
STOPPED_SPEED = 0.5

# Illustrative fuel/emissions proxy coefficients (for *relative* comparison only,
# not calibrated units): a base idle burn, a cruising term (~drag/rolling, grows
# with speed), and a traction term charged only while accelerating (~power, v*a).
# Idling and stop-and-go therefore cost fuel, which is the behaviour we want to
# penalise.
FUEL_IDLE = 0.2
FUEL_CRUISE = 0.02
FUEL_ACCEL = 0.05


@dataclass
class StepMetrics:
    """A single time-step's aggregate flow measurements.

    One of these is appended to the collector's history after every
    :meth:`TrafficSim.step`. Together they form the time series plotted by the
    visualiser and the intended training signal for the flow-prediction model.
    """

    t: float
    mean_speed: float       # mean speed over all cars [m/s]
    n_stopped: int          # cars at/below STOPPED_SPEED (queue length)
    n_crossings: int        # intersection crossings during this step (throughput)


@dataclass
class TripMetrics:
    """One completed destination-leg: how the journey went versus the ideal.

    A "trip" is a single origin -> destination leg (from when a destination is
    assigned to when it is reached). ``delay`` — the excess over the uncongested
    free-flow time — is the headline quantity: it is what signals, priority and
    (later) a learned policy actually reduce. Only produced under a
    destination-aware router (:class:`~traffic_sim.routing.ShortestPathRouter`);
    a wandering router yields none.
    """

    car_id: int
    start_t: float          # when the destination was assigned [s]
    end_t: float            # when it was reached [s]
    travel_time: float      # end_t - start_t [s]
    free_flow_time: float   # ideal uncongested time for this leg [s]
    delay: float            # travel_time - free_flow_time [s]
    stops: int              # number of times the car came to a halt en route
    stopped_time: float     # total time spent at/below STOPPED_SPEED [s]


@dataclass
class MetricsCollector:
    """Records flow diagnostics as the simulation runs, from the outside.

    Injected into :class:`TrafficSim` (which calls :meth:`record` each step) or
    driven manually. It keeps a per-step ``history`` of :class:`StepMetrics`,
    plus cumulative ``edge_crossings`` (a throughput map). ``stopped_speed`` is
    the speed at or below which a car counts as queued. The collector never
    mutates the simulation — it only observes — so the core stays unaware of it.
    """

    stopped_speed: float = STOPPED_SPEED
    history: List[StepMetrics] = field(default_factory=list)
    # Per-edge cumulative crossings (cars that left that edge into a new one).
    edge_crossings: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # Completed destination-legs (empty under a wandering router).
    trips: List[TripMetrics] = field(default_factory=list)
    # Safety / emissions. ``crashes`` mirrors the simulation's own count of
    # genuine collisions (a car that could not stop even at its physical braking
    # limit); ``max_decel`` is the peak deceleration observed (bounded by the
    # cars' ``max_brake`` except at a crash impulse) — a useful sanity check.
    crashes: int = 0            # genuine collisions (from TrafficSim.crashes)
    max_decel: float = 0.0      # peak observed deceleration [m/s^2]
    fuel_proxy: float = 0.0     # cumulative fuel/emissions surrogate [proxy units]
    # Per-edge time series (opt-in): one ``{edge_id: (count, mean_speed)}``
    # snapshot per step, for occupied edges only — the spatiotemporal substrate
    # the flow-prediction model trains on. Off by default to avoid the overhead.
    record_edges: bool = False
    edge_history: List[Dict[int, Tuple[int, float]]] = field(default_factory=list)
    # Per-intersection throughput (cars crossing each node) and cumulative wait
    # (stopped vehicle-seconds on that node's approaches) — localizes congestion.
    node_crossings: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    node_wait: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    _last_edge: Dict[int, int] = field(default_factory=dict)
    _last_v: Dict[int, float] = field(default_factory=dict)
    # Per-car in-progress trip state.
    _trip: Dict[int, dict] = field(default_factory=dict)
    _last_t: Optional[float] = None
    # Crashes recorded before the last reset() — the sim's own counter is
    # cumulative for the whole run, so ``crashes`` reports it minus this base.
    _crash_base: int = 0

    def reset(self) -> None:
        """Forget everything recorded so far and start a fresh window.

        Everything reported afterwards — time series, trips, crashes, fuel,
        per-node waits — covers only what happens *after* the reset, so the
        collector can measure a new parameter regime cleanly (e.g. after a
        dashboard knob was turned). The simulation itself is untouched.
        In-progress trips are dropped and restart from each car's current
        position on the next :meth:`record`, so their delay baselines are
        consistent with the new regime; the sim's cumulative crash counter is
        re-baselined so ``crashes`` counts collisions since the reset.
        """
        self._crash_base += self.crashes
        self.history.clear()
        self.edge_crossings.clear()
        self.trips.clear()
        self.crashes = 0
        self.max_decel = 0.0
        self.fuel_proxy = 0.0
        self.edge_history.clear()
        self.node_crossings.clear()
        self.node_wait.clear()
        self._last_edge.clear()
        self._last_v.clear()
        self._trip.clear()
        self._last_t = None

    def record(self, sim) -> None:
        """Snapshot ``sim``'s current state into a new :class:`StepMetrics`.

        Computes mean speed and queue length over all cars, and counts
        intersection crossings by detecting which cars changed edge since the
        previous call (also accumulating per-edge crossings in
        ``edge_crossings``). From per-car speed deltas it tracks peak
        deceleration and a fuel/emissions proxy, and mirrors the simulation's
        genuine-collision count. Also advances per-car trip tracking, appending a
        :class:`TripMetrics` whenever a car reaches its destination. Call once
        per step, after the step has advanced.
        """
        t = sim.t
        dt = (t - self._last_t) if self._last_t is not None else 0.0
        # Road aggregates cover cars actually in the flow; parked (inactive) cars
        # are off the road and excluded. Trip tracking below still sees them all,
        # so a car parking counts as an arrival.
        cars = [c for c in sim.cars if getattr(c, "active", True)]

        speeds = [c.v for c in cars]
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        n_stopped = sum(1 for v in speeds if v <= self.stopped_speed)

        crossings = 0
        for c in cars:
            prev = self._last_edge.get(c.id)
            if prev is not None and prev != c.edge_id:
                crossings += 1
                self.edge_crossings[prev] += 1
                self.node_crossings[sim.net.edges[prev].v] += 1  # crossed this node
            self._last_edge[c.id] = c.edge_id

            # Time spent stopped counts as wait at the node this edge leads into.
            if dt > 0 and c.v <= self.stopped_speed:
                self.node_wait[sim.net.edges[c.edge_id].v] += dt

            # Peak deceleration (a sanity check on the physical brake cap) and a
            # fuel/emissions proxy, from this car's acceleration since last step.
            v_prev = self._last_v.get(c.id)
            if v_prev is not None and dt > 0:
                a = (c.v - v_prev) / dt          # signed [m/s^2]
                if -a > self.max_decel:
                    self.max_decel = -a
                self.fuel_proxy += (FUEL_IDLE + FUEL_CRUISE * c.v
                                    + FUEL_ACCEL * c.v * max(0.0, a)) * dt
            self._last_v[c.id] = c.v

        # The simulation counts genuine collisions (couldn't stop at max
        # braking) cumulatively; report them relative to the last reset().
        sim_crashes = getattr(sim, "crashes", None)
        if sim_crashes is not None:
            self.crashes = sim_crashes - self._crash_base

        # A destination-aware router exposes free_flow_time; without it (a
        # wandering router) cars have no destination and no trips are produced.
        free_flow = getattr(sim.router, "free_flow_time", None)
        for c in sim.cars:                       # all cars, so parking = arrival
            self._update_trip(c, sim, t, dt, free_flow)

        if self.record_edges:
            acc: Dict[int, Tuple[int, float]] = {}
            for c in cars:
                cnt, ssum = acc.get(c.edge_id, (0, 0.0))
                acc[c.edge_id] = (cnt + 1, ssum + c.v)
            self.edge_history.append(
                {e: (cnt, ssum / cnt) for e, (cnt, ssum) in acc.items()})

        self.history.append(StepMetrics(t, mean_speed, n_stopped, crossings))
        self._last_t = t

    def _open_trip(self, car, sim, t: float, target: int, free_flow) -> dict:
        """Start tracking a new leg toward ``target`` from the car's position now.

        The baseline is the free-flow time from the car's *current position*
        (remaining distance on this edge + cost-to-go from the node ahead) to
        ``target``, so it is consistent with ``start_t = t`` and arrival being
        detected when the car later leaves ``target``.
        """
        baseline = None
        if free_flow is not None:
            edge = sim.net.edges[car.edge_id]
            ahead = free_flow(edge.v, target)
            if ahead != math.inf and edge.speed_limit > 0:
                baseline = (edge.length - car.s) / edge.speed_limit + ahead
        return {"target": target, "start_t": t, "free_flow": baseline,
                "stops": 0, "stopped_time": 0.0,
                "was_stopped": car.v <= self.stopped_speed, "armed": False}

    def _finalize_trip(self, car, state, t: float) -> None:
        """Append a completed :class:`TripMetrics` from ``state`` (skips a trip
        whose free-flow baseline is unknown)."""
        if state["free_flow"] is None:
            return
        travel = t - state["start_t"]
        self.trips.append(TripMetrics(
            car_id=car.id, start_t=state["start_t"], end_t=t,
            travel_time=travel, free_flow_time=state["free_flow"],
            # Clamp at 0: an edge-point stops just before its node, so the
            # node-based baseline can marginally exceed the actual time.
            delay=max(0.0, travel - state["free_flow"]),
            stops=state["stops"], stopped_time=state["stopped_time"]))

    def _update_trip(self, car, sim, t: float, dt: float, free_flow) -> None:
        """Advance one car's trip tracking for this step.

        A trip completes on **arrival**. Under park-and-dwell that is the moment
        the car goes inactive (parks at its destination). Otherwise (sail-through)
        it is when the car reaches an edge *leaving* its target node (``edge.u ==
        target``) — the router reassigns the destination one edge early, so
        watching ``car.dest`` would end the trip too soon, and the ``armed`` guard
        stops a freshly opened trip completing instantly. On arrival, finalise the
        leg and open the next one; while parked, no trip is tracked (so dwell time
        never counts as travel time). Between boundaries, accumulate stops and
        stopped time.
        """
        state = self._trip.get(car.id)

        if not getattr(car, "active", True):
            # Parked: the trip in progress just completed on arrival.
            if state is not None:
                self._finalize_trip(car, state, t)
                self._trip[car.id] = None
            return

        if state is None:
            if car.dest is not None:  # begin (or restart after waking) a trip
                self._trip[car.id] = self._open_trip(car, sim, t, car.dest, free_flow)
            return

        cur_u = sim.net.edges[car.edge_id].u
        if not state["armed"]:
            if cur_u != state["target"]:
                state["armed"] = True
        elif cur_u == state["target"] and car.dest is not None:
            self._finalize_trip(car, state, t)
            self._trip[car.id] = self._open_trip(car, sim, t, car.dest, free_flow)
            return

        stopped = car.v <= self.stopped_speed
        if stopped:
            state["stopped_time"] += dt
            if not state["was_stopped"]:
                state["stops"] += 1
        state["was_stopped"] = stopped

    def occupancy(self, sim) -> Dict[int, int]:
        """Current car count per edge id (a density proxy)."""
        counts: Dict[int, int] = defaultdict(int)
        for c in sim.cars:
            counts[c.edge_id] += 1
        return dict(counts)

    def edge_count_series(self, edge_id: int) -> List[int]:
        """Per-step car count on ``edge_id`` (0 when empty). Needs ``record_edges``."""
        return [snap.get(edge_id, (0, 0.0))[0] for snap in self.edge_history]

    def edge_speed_series(self, edge_id: int) -> List[float]:
        """Per-step mean speed on ``edge_id`` [m/s] (0.0 when empty)."""
        return [snap.get(edge_id, (0, 0.0))[1] for snap in self.edge_history]

    def edge_density_series(self, edge_id: int, net) -> List[float]:
        """Per-step density on ``edge_id`` [veh/m] = count / edge length."""
        length = net.edges[edge_id].length
        return [c / length for c in self.edge_count_series(edge_id)]

    def node_mean_wait(self, node_id: int) -> float:
        """Mean stopped time [s] per vehicle crossing ``node_id`` (0 if none)."""
        n = self.node_crossings.get(node_id, 0)
        return self.node_wait.get(node_id, 0.0) / n if n else 0.0

    def busiest_nodes(self, k: int = 5) -> List[Tuple[int, float, int]]:
        """The ``k`` most-congested nodes, worst first, as
        ``(node_id, total_wait_s, crossings)`` by cumulative wait."""
        items = [(nid, w, self.node_crossings.get(nid, 0))
                 for nid, w in self.node_wait.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]

    def fundamental_samples(self, net) -> List[Tuple[float, float, float]]:
        """All ``(density [veh/km], flow [veh/h], speed [km/h])`` samples across
        every occupied edge and step — the points of the fundamental diagram.

        Flow uses the hydrodynamic relation ``q = k * v`` (density times mean
        speed). Needs ``record_edges=True``.
        """
        samples: List[Tuple[float, float, float]] = []
        for snap in self.edge_history:
            for eid, (count, speed) in snap.items():
                k = count / net.edges[eid].length * 1000.0     # veh/km
                v = speed * 3.6                                # km/h
                samples.append((k, k * v, v))                  # q = k*v  [veh/h]
        return samples

    @property
    def times(self) -> List[float]:
        """The recorded time stamps [s], one per step, in order."""
        return [m.t for m in self.history]

    @property
    def mean_speeds(self) -> List[float]:
        """The per-step mean car speed [m/s] time series."""
        return [m.mean_speed for m in self.history]

    @property
    def queue_lengths(self) -> List[int]:
        """The per-step queue length (count of stopped cars) time series."""
        return [m.n_stopped for m in self.history]

    @property
    def delays(self) -> List[float]:
        """Per-completed-trip delay [s] (travel time minus free-flow time)."""
        return [tp.delay for tp in self.trips]

    def summary(self) -> dict:
        """Aggregate the whole run into a small dict of scalar statistics.

        Includes step count, wall-clock duration, average and peak queue,
        average speed, total intersection crossings, and throughput per second.
        When completed trips exist (a destination-aware router), also includes
        trip-level travel time, **delay**, stops per trip, and a fairness ratio
        (p95 delay over mean delay — how much worse the unlucky tail fares).
        Always includes the safety (crashes, peak deceleration) and fuel-proxy
        totals. Returns ``{"steps": 0}`` if nothing was recorded.
        """
        if not self.history:
            return {"steps": 0}
        n = len(self.history)
        total_crossings = sum(m.n_crossings for m in self.history)
        duration = self.history[-1].t - self.history[0].t
        out = {
            "steps": n,
            "duration_s": duration,
            "avg_speed": sum(self.mean_speeds) / n,
            "avg_queue": sum(self.queue_lengths) / n,
            "max_queue": max(self.queue_lengths),
            "total_crossings": total_crossings,
            "throughput_per_s": total_crossings / duration if duration > 0 else 0.0,
            "crashes": self.crashes,
            "max_decel": self.max_decel,
            "fuel_proxy": self.fuel_proxy,
            "intersection_wait_s": sum(self.node_wait.values()),
            "mean_wait_per_crossing_s": (
                sum(self.node_wait.values()) / total_crossings
                if total_crossings else 0.0),
        }
        if self.trips:
            delays = sorted(self.delays)
            m = len(delays)
            mean_delay = sum(delays) / m
            p95 = _percentile(delays, 0.95)
            out.update({
                "trips_completed": m,
                "mean_travel_time_s": sum(tp.travel_time for tp in self.trips) / m,
                "mean_delay_s": mean_delay,
                "median_delay_s": _percentile(delays, 0.50),
                "p90_delay_s": _percentile(delays, 0.90),
                "p95_delay_s": p95,
                "mean_stops_per_trip": sum(tp.stops for tp in self.trips) / m,
                # Fairness: tail delay relative to the mean (1.0 = perfectly even).
                "delay_p95_over_mean": (p95 / mean_delay) if mean_delay > 0 else 0.0,
            })
        return out


def _percentile(sorted_values: List[float], p: float) -> float:
    """The ``p`` quantile (0..1) of an already-sorted list (nearest-rank)."""
    if not sorted_values:
        return 0.0
    k = min(len(sorted_values) - 1, int(p * len(sorted_values)))
    return sorted_values[k]
