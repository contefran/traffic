"""Vehicle model.

A car's spatial state is one-dimensional: which edge it is on (``edge_id``)
and how far along that edge it has travelled (``s``). The remaining fields are
the car-following parameters used by the simulation.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Car:
    id: int
    edge_id: int
    s: float        # position along edge, in [0, edge.length]
    v: float        # speed [m/s]

    # Physical / behavioural parameters (per-car so they can be varied).
    max_speed: float = 50.0       # [m/s] hard cap on desired speed
    length: float = 4.5           # [m]
    accel: float = 2.0            # [m/s^2]
    braking: float = 4.0          # [m/s^2]
    s0: float = 2.0               # minimum standstill gap [m]
    time_headway: float = 1.2     # desired time gap to leader [s]

    # History of (t, edge_id, s) samples, for debugging / metrics.
    trail: deque = field(default_factory=lambda: deque(maxlen=200))
