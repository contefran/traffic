"""Activity-based daily schedules — each car is an agent with its own routine.

This is the richer successor to :mod:`traffic_sim.schedule`: instead of an
aggregate origin→destination matrix, **every car gets a full, personal daily
plan** — home → work (maybe a lunch out) → home, with an optional evening of
gym / dining / pubs, jittered times, and a "maybe be late" tail. The plan is
computed **once for the whole day**, is **periodic** (it repeats every day,
seeded so identically), and is then executed starting at **exactly midnight** —
so a car is placed wherever its own schedule says it is at 00:00. A commuter home
by evening starts the sim asleep at home; a night-owl still at a pub at 2 am
starts the sim *at the pub*, drives home, and is back at the pub next midnight.
Night-owls therefore **emerge from the schedules** rather than a tunable fraction.

Venues (restaurants / gyms / pubs) are a **shadow overlay**, not a land-use zone:
they sit on edges of every zone, sampled retail-leaning (a pub can be anywhere,
but shops/restaurants cluster on retail streets). Lunch is taken at the
restaurant nearest a car's workplace.

The model reuses the simulation's park/wake machinery: a car parks at each leg's
destination and its wake time is simply the *next* leg's departure, so the dwell
(a work day, an hour at lunch, a night's sleep) emerges from the plan's times.
Pluggable like the other models (``schedule=`` on
:class:`~traffic_sim.simulation.TrafficSim`).
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .zones import LandUse, edges_by_zone


class ActivityKind(Enum):
    """What a car is doing at a leg's destination (for readability / rendering)."""

    HOME = "home"
    WORK = "work"
    LUNCH = "lunch"
    GYM = "gym"
    DINE = "dine"
    PUB = "pub"


@dataclass
class Activity:
    """One leg of a daily plan: leave at ``depart`` for ``dest_edge`` (an address
    ``dest_frac`` along it) to do ``kind``. ``depart`` is sim-seconds in [0, day)."""

    depart: float
    dest_edge: int
    dest_frac: float
    kind: ActivityKind


# How likely each optional activity is (per car, per day).
P_LUNCH_OUT = 0.40   # leave the office for lunch (else eat at desk)
P_GYM = 0.20
P_DINE_OUT = 0.25
P_PUB = 0.20
P_SECOND_PUB = 0.40  # given a first pub


def _edge_mid(net, eid):
    """The (x, y) midpoint of edge ``eid``."""
    e = net.edges[eid]
    a, b = net.nodes[e.u], net.nodes[e.v]
    return (a.x + b.x) / 2.0, (a.y + b.y) / 2.0


class Venues:
    """Restaurant / gym / pub edges (a retail-leaning overlay over all zones)."""

    def __init__(self, restaurants, gyms, pubs):
        """Store the three venue-edge pools as lists (for random choice)."""
        self.restaurants = list(restaurants)
        self.gyms = list(gyms)
        self.pubs = list(pubs)


def assign_venues(net, zones, *, seed: int = 0, restaurant_frac: float = 0.10,
                  gym_frac: float = 0.03, pub_frac: float = 0.05,
                  retail_weight: float = 3.0) -> Venues:
    """Sample venue edges across the whole city, **leaning toward retail streets**.

    Each zonable street may host a venue of each kind with probability
    ``*_frac`` (times ``retail_weight`` on a retail street). Venues are an overlay:
    a street keeps its land use and can also be a restaurant/gym/pub. Seeded.
    """
    rng = random.Random(seed)
    ground = [eid for eid in zones]                # zonable ground streets

    def sample(frac):
        """Pick venue edges at rate ``frac`` (x``retail_weight`` on retail streets)."""
        out = []
        for eid in ground:
            w = retail_weight if zones.get(eid) is LandUse.RETAIL else 1.0
            if rng.random() < frac * w:
                out.append(eid)
        return out or ground[:1]                   # never empty

    return Venues(sample(restaurant_frac), sample(gym_frac), sample(pub_frac))


class ActivitySchedule:
    """Per-car daily itineraries (see module docs)."""

    def __init__(self, venues: Venues, *, day_length: float = 600.0,
                 seed: int = 0, work_scale: float = 400.0) -> None:
        """``venues`` supplies the restaurant/gym/pub pools; ``day_length`` is the
        simulated seconds in a day (plan times are placed on it). ``work_scale``
        [m] is the Gaussian distance scale for choosing a workplace **near home**
        — people cluster their home and job in a big city, so a job is picked with
        probability ``exp(-d²/2·work_scale²)`` in the home→office distance ``d``,
        which keeps commutes realistic instead of criss-crossing the whole map.
        Seeded so every car's plan is reproducible."""
        self.venues = venues
        self.day_length = day_length
        self.rng = random.Random(seed)
        self.work_scale = work_scale
        self.net = None   # set in :meth:`assign`

    # ---- helpers -------------------------------------------------------------

    def _h(self, hour: float) -> float:
        """Clock ``hour`` (may exceed 24 for after-midnight) -> sim-seconds mod day."""
        return (self.day_length * (hour / 24.0)) % self.day_length

    def _next_time(self, depart: float, t: float) -> float:
        """Next sim-time at/after ``t`` whose time-of-day is ``depart``."""
        wake = int(t // self.day_length) * self.day_length + depart
        if wake < t - 1e-9:
            wake += self.day_length
        return wake

    def _nearest(self, work_edge: int, pool: List[int]) -> int:
        """The pool edge whose midpoint is closest to ``work_edge``'s midpoint."""
        wx, wy = _edge_mid(self.net, work_edge)
        return min(pool, key=lambda e: (lambda m: (m[0] - wx) ** 2 + (m[1] - wy) ** 2)
                   (_edge_mid(self.net, e)))

    def _frac(self) -> float:
        """A random mid-block address fraction."""
        return self.rng.uniform(0.3, 0.9)

    # ---- plan generation -----------------------------------------------------

    def _make_plan(self, car) -> List[Activity]:
        """Build one car's periodic day as a sequence of legs (home ends the day)."""
        rng = self.rng
        work_frac = self._frac()
        home_frac = self._frac()
        lunch_rest = self._nearest(car.work, self.venues.restaurants)

        legs: List[tuple] = []   # (clock_hour, dest_edge, dest_frac, kind)
        wake = rng.triangular(5.0, 9.5, 7.0)          # wakes ~07:00, some outliers
        legs.append((wake + rng.uniform(0.5, 1.0), car.work, work_frac, ActivityKind.WORK))

        if rng.random() < P_LUNCH_OUT:                # an hour out for lunch
            lunch = rng.uniform(12.0, 13.0)
            legs.append((lunch, lunch_rest, self._frac(), ActivityKind.LUNCH))
            legs.append((lunch + rng.uniform(0.75, 1.25), car.work, work_frac,
                         ActivityKind.WORK))

        t = rng.triangular(16.5, 19.0, 17.5)          # leaves work (maybe late)
        if rng.random() < P_GYM:
            legs.append((t, rng.choice(self.venues.gyms), self._frac(), ActivityKind.GYM))
            t += rng.uniform(1.0, 1.5)
        if rng.random() < P_DINE_OUT:
            legs.append((t, rng.choice(self.venues.restaurants), self._frac(),
                         ActivityKind.DINE))
            t += rng.uniform(1.0, 1.5)
        if rng.random() < P_PUB:
            t = max(t, rng.uniform(20.0, 22.0))       # pubs get going later
            legs.append((t, rng.choice(self.venues.pubs), self._frac(), ActivityKind.PUB))
            t += rng.uniform(1.0, 2.0)
            if rng.random() < P_SECOND_PUB:
                legs.append((t, rng.choice(self.venues.pubs), self._frac(),
                             ActivityKind.PUB))
                t += rng.uniform(1.0, 2.0)

        legs.append((t + rng.uniform(0.2, 0.6), car.home, home_frac, ActivityKind.HOME))
        return [Activity(self._h(h), e, f, k) for (h, e, f, k) in legs]

    def _place_at_midnight(self, car) -> None:
        """Put ``car`` wherever its plan says it is at 00:00 (parked), asleep until
        the next departure. The active leg at midnight is the last one to depart
        before it (the largest ``depart``, since later legs have wrapped past 24h)."""
        plan = car.plan
        idx = max(range(len(plan)), key=lambda i: plan[i].depart)
        leg = plan[idx]
        edge = self.net.edges[leg.dest_edge]
        car.edge_id, car.lane = leg.dest_edge, 0
        car.s = leg.dest_frac * edge.length
        car.v = 0.0
        car.active = False
        car.plan_idx = idx
        car.dest = car.dest_edge = car.next_edge = None
        nxt = plan[(idx + 1) % len(plan)]
        car.wake_t = self._next_time(nxt.depart, 0.0)

    # ---- setup + runtime -----------------------------------------------------

    def _pick_work(self, car, offices, office_mid) -> int:
        """An office for ``car``, weighted toward those near its home (Gaussian)."""
        hx, hy = _edge_mid(self.net, car.home)
        s2 = 2.0 * self.work_scale * self.work_scale
        weights = [math.exp(-((office_mid[e][0] - hx) ** 2
                              + (office_mid[e][1] - hy) ** 2) / s2) for e in offices]
        if sum(weights) <= 0.0:                       # all far (numerical underflow)
            return min(offices, key=lambda e: (office_mid[e][0] - hx) ** 2
                       + (office_mid[e][1] - hy) ** 2)
        return self.rng.choices(offices, weights=weights, k=1)[0]

    def assign(self, cars, net, zones) -> None:
        """Give each car a workplace (near home) and a periodic plan, and place it
        at midnight."""
        self.net = net
        offices = edges_by_zone(zones).get(LandUse.OFFICE) or list(zones) or [0]
        office_mid = {e: _edge_mid(net, e) for e in offices}
        for car in cars:
            car.work = self._pick_work(car, offices, office_mid)
            car.plan = self._make_plan(car)
            self._place_at_midnight(car)

    def on_wake(self, car, t: float) -> None:
        """A parked car's dwell is over: advance to the next leg and head there."""
        car.plan_idx = (car.plan_idx + 1) % len(car.plan)
        leg = car.plan[car.plan_idx]
        car.dest_edge = leg.dest_edge
        car.dest = self.net.edges[leg.dest_edge].u   # route to the street's upstream node
        car.dest_frac = leg.dest_frac

    def on_park(self, car, t: float) -> None:
        """A car has arrived at this leg's destination: sleep until the next leg."""
        nxt = car.plan[(car.plan_idx + 1) % len(car.plan)]
        car.wake_t = self._next_time(nxt.depart, t)
