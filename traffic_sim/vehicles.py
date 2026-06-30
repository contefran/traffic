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
    id: int
    edge_id: int
    s: float        # position along edge, in [0, edge.length]
    v: float        # speed [m/s]

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
    braking: float = 4.0          # [m/s^2]
    s0: float = 2.0               # minimum standstill gap [m]
    time_headway: float = 1.2     # desired time gap to leader [s]

    # History of (t, edge_id, s) samples, for debugging / metrics.
    trail: deque = field(default_factory=lambda: deque(maxlen=200))
