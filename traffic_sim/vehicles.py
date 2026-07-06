"""Vehicle model.

A car's spatial state is one-dimensional: which edge it is on (``edge_id``)
and how far along that edge it has travelled (``s``). The remaining fields are
the car-following parameters used by the simulation.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Car:
    """A single vehicle: 1-D state on the road graph plus its driver parameters.

    Spatial state is minimal — the ``edge_id`` it occupies, the distance ``s``
    travelled along that edge, and the speed ``v``. World position is derived on
    demand via :meth:`RoadNetwork.point_on_edge`, never stored here. The
    car-following parameters (``accel``, ``braking``, ``s0``, ``time_headway``,
    ``max_speed``, ``length``) are per-car so heterogeneous drivers can be mixed;
    the Intelligent Driver Model in :meth:`TrafficSim.step` reads them directly.
    """

    id: int
    edge_id: int
    s: float        # position along edge, in [0, edge.length]
    v: float        # speed [m/s]
    lane: int = 0   # lane index on the current edge, 0 = rightmost

    # The edge this car will move onto at the end of the current one. Committed
    # in advance (before the stop line) so the signal can gate the specific
    # movement; ``None`` until the router decides. Reset to None on each transfer.
    next_edge: Optional[int] = None

    # Destination node id for destination-based routing. ``None`` means "no
    # destination" (the car wanders, e.g. under RandomRouter). A
    # destination-aware router reads this to steer toward ``dest`` and assigns
    # a fresh one when the car arrives.
    dest: Optional[int] = None
    # The specific destination **street** (edge id) the car is headed for, when
    # routing edge-precisely (``dest`` is then that edge's *upstream* node, the
    # router's target; the car is forced onto ``dest_edge`` for the final hop and
    # stops ``dest_frac`` along it). ``None`` falls back to node-based arrival
    # (park on any edge whose ``v == dest``).
    dest_edge: Optional[int] = None
    # Where along the destination edge the address actually is, as a fraction in
    # (0, 1]: 1.0 = the far intersection, <1 = a point mid-block (an address on
    # the street). Only used with park-and-dwell.
    dest_frac: float = 1.0
    # The car's home: a residential street (edge id) it returns to whenever a trip
    # is residential-bound. ``None`` = no fixed home (residential trips then go to
    # a random residential street).
    home: Optional[int] = None

    # Vehicle class name (see :data:`DEFAULT_FLEET`), for rendering / stats only —
    # the dynamics read the physical fields below, which the type presets set.
    vtype: str = "city"

    # Physical / behavioural parameters (per-car so they can be varied).
    max_speed: float = 50.0       # [m/s] hard cap on desired speed
    length: float = 4.5           # [m]
    accel: float = 2.0            # [m/s^2]
    braking: float = 4.0          # [m/s^2] comfortable decel (IDM gap planning)
    max_brake: float = 9.0        # [m/s^2] physical decel limit (tyre-road grip)
    s0: float = 2.0               # minimum standstill gap [m]
    time_headway: float = 1.2     # desired time gap to leader [s]

    # Trip lifecycle for park-and-dwell. ``active`` False means the car has
    # arrived and is parked (off the road, not part of the flow); it re-enters
    # when the simulation clock reaches ``wake_t``.
    active: bool = True
    wake_t: float = 0.0
    # This agent's morning departure time (sim-seconds within a day), set by a
    # :class:`~traffic_sim.schedule.DailySchedule`. When the car parks at its own
    # home the sim sleeps it until the next occurrence of this time.
    depart_time: float = 0.0
    # Activity-based scheduling (:mod:`traffic_sim.activities`): ``work`` is the
    # car's fixed workplace street; ``plan`` is its periodic daily itinerary (a
    # list of :class:`~traffic_sim.activities.Activity` legs) and ``plan_idx`` the
    # leg it is currently at / heading to.
    work: Optional[int] = None
    plan: Optional[list] = None
    plan_idx: int = 0

    # History of (t, edge_id, s) samples, for debugging / metrics.
    trail: deque = field(default_factory=lambda: deque(maxlen=200))


@dataclass(frozen=True)
class VehicleType:
    """A class of vehicle: its size and performance envelope.

    The fields map **straight onto the per-car IDM parameters**, so the dynamics
    need no special-casing — a truck is simply longer, slower to accelerate, worse
    at braking, and keeps a bigger gap. Slow heavy vehicles then get overtaken by
    faster ones through the existing MOBIL lane-change model, for free.
    """

    name: str
    length: float         # [m]
    max_speed: float      # [m/s] top speed (binds on fast roads)
    accel: float          # [m/s^2] comfortable acceleration
    braking: float        # [m/s^2] comfortable deceleration (IDM gap planning)
    max_brake: float      # [m/s^2] physical braking limit (worse for heavy vehicles)
    s0: float             # [m] standstill gap
    time_headway: float   # [s] desired time gap to the leader


# The default fleet: (share, type). Mostly small city cars, with a few sports
# cars (fast, aggressive), trucks and buses (long, slow to accelerate, poor
# braking, big gaps) — enough of a mix to produce overtaking and platooning
# behind slow heavy vehicles.
DEFAULT_FLEET: List[Tuple[float, VehicleType]] = [
    (0.78, VehicleType("city",  4.5, 50.0, 2.0, 4.0, 9.0,  2.0, 1.2)),
    (0.08, VehicleType("sport", 4.2, 60.0, 3.6, 5.0, 10.5, 1.6, 0.9)),
    (0.09, VehicleType("truck", 14.0, 20.0, 0.7, 3.0, 6.5, 3.0, 1.8)),
    (0.05, VehicleType("bus",   12.0, 18.0, 0.9, 3.5, 6.5, 2.5, 1.6)),
]


def apply_vehicle_type(car: "Car", vt: VehicleType, rng: Optional[random.Random] = None,
                       jitter: float = 0.1) -> None:
    """Set ``car``'s size/performance fields from vehicle type ``vt``.

    ``length``, ``max_speed`` and ``max_brake`` are the type's exactly; the
    behavioural gaps (``accel``, ``braking``, ``s0``, ``time_headway``) get a
    small per-driver ``jitter`` (±10% by default) when an ``rng`` is given, so no
    two drivers of the same vehicle are identical — a light stand-in for driver
    personality without a separate model.
    """
    car.vtype = vt.name
    car.length = vt.length
    car.max_speed = vt.max_speed
    car.max_brake = vt.max_brake
    j = (lambda x: x) if rng is None else (lambda x: x * rng.uniform(1 - jitter, 1 + jitter))
    car.accel = j(vt.accel)
    car.braking = j(vt.braking)
    car.s0 = j(vt.s0)
    car.time_headway = j(vt.time_headway)


def assign_vehicle_types(cars, *, seed: int = 0, fleet=DEFAULT_FLEET,
                         jitter: float = 0.1) -> None:
    """Give each car a vehicle type drawn from ``fleet``'s share distribution.

    Deterministic under ``seed``; applies each type's parameters (with a small
    per-driver ``jitter`` on the behavioural gaps). See :func:`apply_vehicle_type`.
    """
    rng = random.Random(seed)
    weights = [w for w, _ in fleet]
    types = [vt for _, vt in fleet]
    for car in cars:
        apply_vehicle_type(car, rng.choices(types, weights=weights, k=1)[0], rng, jitter)
