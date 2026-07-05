"""Time-of-day travel demand: where trips go, and when.

A :class:`DemandModel` turns the land-use zones (:mod:`traffic_sim.zones`) into
realistic, *non-stationary* origin→destination flows. A day is split into four
periods; in each, an origin→destination **zone** matrix says where a trip
starting in a given zone is likely to end up — home→work in the morning,
work→home in the evening, errands to retail at midday, home at night. The model
picks a destination *zone* from that matrix, then a random node in it.

Plugged into :class:`~traffic_sim.routing.ShortestPathRouter` (``demand=``): the
router still finds the fastest path, the demand model only chooses the endpoint.
Because it depends on the time of day, the simulation feeds the router the
current time each step (``router.now``). ``demand=None`` keeps the old uniform
random destinations. Non-stationary demand is exactly the richer training signal
the later flow-prediction model wants.
"""

import random
from enum import Enum
from typing import Dict, Optional

from .zones import LandUse, ZoneMap, nodes_by_zone


class Period(Enum):
    """The four parts of the simulated day, each with its own travel pattern."""

    MORNING = "morning"   # home -> work
    MIDDAY = "midday"     # errands, retail
    EVENING = "evening"   # work -> home
    NIGHT = "night"       # home, quiet


# Origin→destination zone weights per period. ``None`` is the fallback applied to
# any origin zone not listed explicitly for that period. Weights need not sum to 1.
_R, _O, _T = LandUse.RESIDENTIAL, LandUse.OFFICE, LandUse.RETAIL
DEFAULT_OD: Dict[Period, Dict[Optional[LandUse], Dict[LandUse, float]]] = {
    Period.MORNING: {
        _R: {_O: 0.8, _T: 0.1, _R: 0.1},          # commute to work
        None: {_O: 0.5, _T: 0.3, _R: 0.2},
    },
    Period.MIDDAY: {
        None: {_T: 0.45, _O: 0.3, _R: 0.25},      # errands / lunch
    },
    Period.EVENING: {
        _O: {_R: 0.8, _T: 0.1, _O: 0.1},          # commute home
        None: {_R: 0.55, _T: 0.3, _O: 0.15},
    },
    Period.NIGHT: {
        None: {_R: 0.7, _T: 0.2, _O: 0.1},        # mostly home
    },
}


class DemandModel:
    """Time-of-day destination chooser driven by land-use zones (see module docs)."""

    def __init__(self, net, zones: ZoneMap, *, seed: int = 0,
                 day_length: float = 600.0, od=None) -> None:
        """``day_length`` is the simulated seconds in one day (the four periods
        split it evenly). ``od`` overrides :data:`DEFAULT_OD`. Seeded for
        reproducible destination choices."""
        self.net = net
        self.zones = zones
        self.by_zone = nodes_by_zone(zones)
        self.rng = random.Random(seed)
        self.day_length = day_length
        self.od = od if od is not None else DEFAULT_OD

    def period(self, t: float) -> Period:
        """Which :class:`Period` the time ``t`` falls in (day wraps around)."""
        frac = (t % self.day_length) / self.day_length
        if frac < 0.25:
            return Period.MORNING
        if frac < 0.5:
            return Period.MIDDAY
        if frac < 0.75:
            return Period.EVENING
        return Period.NIGHT

    def _weights(self, t: float, origin_zone: LandUse) -> Dict[LandUse, float]:
        """Destination-zone weights for a trip from ``origin_zone`` at time ``t``."""
        table = self.od[self.period(t)]
        return table.get(origin_zone) or table.get(None) or {origin_zone: 1.0}

    def _pick_zone(self, weights: Dict[LandUse, float]) -> LandUse:
        """Weighted random destination zone."""
        total = sum(weights.values())
        r = self.rng.random() * total
        acc = 0.0
        for zone, w in weights.items():
            acc += w
            if r < acc:
                return zone
        return next(iter(weights))

    def next_destination(self, origin: int, t: float) -> int:
        """A destination node for a trip leaving ``origin`` at time ``t``.

        Samples a destination zone from the time-of-day matrix, then a random
        node in that zone (never ``origin``). Falls back to any node if the chosen
        zone happens to be empty.
        """
        origin_zone = self.zones.get(origin, LandUse.OTHER)
        dest_zone = self._pick_zone(self._weights(t, origin_zone))
        candidates = self.by_zone.get(dest_zone) or [n.id for n in self.net.nodes]
        dest = self.rng.choice(candidates)
        if dest == origin and len(candidates) > 1:
            while dest == origin:
                dest = self.rng.choice(candidates)
        return dest
