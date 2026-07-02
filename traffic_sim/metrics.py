"""Diagnostics: record traffic-flow quantities as the simulation runs.

These are the fundamental variables of traffic flow — speed, density (here as
occupancy/queueing), and throughput — and the intended training signal for the
later flow-prediction model. The collector is dependency-free and observes the
simulation from the outside, so the simulation core stays unaware of it.

Inject one into ``TrafficSim`` (it calls ``record(sim)`` after each step) or
call ``record(sim)`` yourself in a custom loop.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

# A car slower than this is treated as queued/stopped [m/s].
STOPPED_SPEED = 0.5


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
    _last_edge: Dict[int, int] = field(default_factory=dict)

    def record(self, sim) -> None:
        """Snapshot ``sim``'s current state into a new :class:`StepMetrics`.

        Computes mean speed and queue length over all cars, and counts
        intersection crossings by detecting which cars changed edge since the
        previous call (also accumulating per-edge crossings in
        ``edge_crossings``). Call once per step, after the step has advanced.
        """
        cars = sim.cars
        speeds = [c.v for c in cars]
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        n_stopped = sum(1 for v in speeds if v <= self.stopped_speed)

        crossings = 0
        for c in cars:
            prev = self._last_edge.get(c.id)
            if prev is not None and prev != c.edge_id:
                crossings += 1
                self.edge_crossings[prev] += 1
            self._last_edge[c.id] = c.edge_id

        self.history.append(StepMetrics(sim.t, mean_speed, n_stopped, crossings))

    def occupancy(self, sim) -> Dict[int, int]:
        """Current car count per edge id (a density proxy)."""
        counts: Dict[int, int] = defaultdict(int)
        for c in sim.cars:
            counts[c.edge_id] += 1
        return dict(counts)

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

    def summary(self) -> dict:
        """Aggregate the whole run into a small dict of scalar statistics.

        Includes step count, wall-clock duration, average and peak queue,
        average speed, total intersection crossings, and throughput per second.
        Returns ``{"steps": 0}`` if nothing was recorded.
        """
        if not self.history:
            return {"steps": 0}
        n = len(self.history)
        total_crossings = sum(m.n_crossings for m in self.history)
        duration = self.history[-1].t - self.history[0].t
        return {
            "steps": n,
            "duration_s": duration,
            "avg_speed": sum(self.mean_speeds) / n,
            "avg_queue": sum(self.queue_lengths) / n,
            "max_queue": max(self.queue_lengths),
            "total_crossings": total_crossings,
            "throughput_per_s": total_crossings / duration if duration > 0 else 0.0,
        }
