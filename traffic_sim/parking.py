"""Park-and-dwell: cars stop at their destination, wait, then set off again.

Without this a car sails through its destination and instantly picks a new one,
so the number of cars on the road is constant. Real trips are discrete: you
drive somewhere, *park for a while* (a work day, a shopping trip, overnight),
then drive somewhere else. Modelling that makes the number of *active* cars rise
and fall over the day — rush-hour peaks, quiet nights — which is exactly the
non-stationary load the flow-prediction model wants to learn.

This small, pluggable model just decides **how long** a car dwells; the
simulation owns the mechanics (parking a car off the road when it arrives, and
waking it when the clock catches up). Injected via ``parking=`` on
:class:`~traffic_sim.simulation.TrafficSim`; ``None`` keeps the old
sail-through behaviour. The dwell can depend on the destination's land use — a
long day at the office, a short stop at the shops — if a zone map is supplied.
"""

import random
from typing import Optional

from .zones import LandUse, ZoneMap

# Default dwell range per land use [s]: office = long (a work day), retail =
# short (an errand), residential = medium (home), other = medium.
DEFAULT_DWELL = {
    LandUse.OFFICE: (120.0, 240.0),
    LandUse.RETAIL: (15.0, 45.0),
    LandUse.RESIDENTIAL: (60.0, 180.0),
    LandUse.OTHER: (30.0, 90.0),
}


class ParkingModel:
    """Chooses how long a car dwells after arriving (see module docs)."""

    def __init__(self, *, seed: int = 0, zones: Optional[ZoneMap] = None,
                 dwell=None, default_dwell=(30.0, 90.0)) -> None:
        """``zones`` (optional) makes the dwell depend on the destination's land
        use via ``dwell`` (defaults to :data:`DEFAULT_DWELL`); without it every
        stop uses ``default_dwell`` (a ``(min, max)`` range). Seeded."""
        self.rng = random.Random(seed)
        self.zones = zones
        self.dwell_by_use = dwell if dwell is not None else DEFAULT_DWELL
        self.default_dwell = default_dwell

    def dwell_time(self, edge_id: int) -> float:
        """A dwell duration [s] for a car parking on street ``edge_id``.

        The land use of the *street parked on* sets the range (a long day at an
        office street, a quick stop on a retail one); ``default_dwell`` applies
        when there is no zone map or the street is unzoned.
        """
        lo, hi = self.default_dwell
        if self.zones is not None:
            use = self.zones.get(edge_id)
            if use in self.dwell_by_use:
                lo, hi = self.dwell_by_use[use]
        return self.rng.uniform(lo, hi)
