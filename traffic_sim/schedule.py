"""Per-car daily schedules — the activity-based demand layer.

Instead of a fixed pool of cars milling around, each car is an **agent with its
own routine**: most start the day *asleep at home* and leave at a personal,
randomized morning time; a minority are *night-active* (already out at midnight,
they head home and start their day later). Because departures are staggered
across the morning, the rush-hour build-up **emerges** from the agents rather
than being scripted — the hallmark of an activity-based travel model.

This model only samples the times and sets the **midnight state**; it reuses the
simulation's existing park/wake machinery (``Car.active`` / ``Car.wake_t``) to run
the day. It also tells the sim when a car that has parked *at its own home* should
next wake — the next morning's departure — so agents sleep through the night
instead of driving around. Pluggable like the other models (``schedule=`` on
:class:`~traffic_sim.simulation.TrafficSim`); ``None`` keeps every car active from
the start (the old behaviour).
"""

import random
from typing import Optional, Sequence, Tuple


class DailySchedule:
    """Assigns each car a departure time + midnight state (see module docs)."""

    def __init__(self, *, day_length: float, seed: int = 0,
                 night_fraction: float = 0.15,
                 depart: Tuple[float, float, float] = (5.5, 7.5, 9.5),
                 night_depart: Tuple[float, float, float] = (9.0, 11.0, 13.0)) -> None:
        """``day_length`` is the simulated seconds per day (departures are placed
        on it). ``night_fraction`` of cars start the day already out on the road.
        ``depart``/``night_depart`` are ``(earliest, peak, latest)`` **clock
        hours** for the two groups' morning departures, sampled per car from a
        triangular distribution. Seeded for reproducibility.
        """
        self.day_length = day_length
        self.rng = random.Random(seed)
        self.night_fraction = night_fraction
        self.depart = depart
        self.night_depart = night_depart

    def _hour_to_t(self, hour: float) -> float:
        """Clock ``hour`` (0–24) as a sim-time offset within the day [s]."""
        return self.day_length * (hour / 24.0)

    def assign(self, cars: Sequence, net) -> None:
        """Give each car a ``depart_time`` and set its state at midnight.

        A **night-active** car keeps its (already spaced) spawn position and stays
        active. Everyone else is sent home to sleep: repositioned onto their home
        street, mid-block, **inactive** until their morning ``depart_time`` (when
        the sim's wake pass re-enters them with a fresh destination). Home cars
        being off-road at t=0 also means the city starts quiet — no cold-start
        collisions — and fills as the morning departures fire.
        """
        ground = [e.id for e in net.edges
                  if net.nodes[e.u].level == 0 and net.nodes[e.v].level == 0] \
            or [e.id for e in net.edges]
        for car in cars:
            night = self.rng.random() < self.night_fraction
            lo, peak, hi = self.night_depart if night else self.depart
            car.depart_time = self._hour_to_t(self.rng.triangular(lo, hi, peak))
            if night:
                car.active, car.wake_t = True, 0.0        # out already; keep spawn spot
                continue
            home = car.home if car.home is not None else self.rng.choice(ground)
            edge = net.edges[home]
            car.edge_id = home
            car.s = self.rng.uniform(0.2, 0.8) * edge.length  # a house mid-street
            car.lane = 0
            car.active, car.wake_t = False, car.depart_time
            car.dest = car.dest_edge = car.next_edge = None

    def next_departure(self, car, t: float) -> float:
        """Next sim-time strictly after ``t`` matching the car's daily departure.

        Used when a car parks *at its home* — it then sleeps until this next
        morning departure instead of taking the short residential dwell.
        """
        wake = int(t // self.day_length) * self.day_length + car.depart_time
        if wake <= t:
            wake += self.day_length
        return wake
