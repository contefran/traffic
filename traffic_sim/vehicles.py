"""Vehicle model.

A car's spatial state is one-dimensional: which edge it is on (``edge_id``)
and how far along that edge it has travelled (``s``). The remaining fields are
the car-following parameters used by the simulation.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


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

    # History of (t, edge_id, s) samples, for debugging / metrics.
    trail: deque = field(default_factory=lambda: deque(maxlen=200))
