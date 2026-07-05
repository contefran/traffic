"""Land-use zones: a spatial overlay tagging each node residential / office / retail.

Kept **decoupled** from the network model (``Node`` stays pure geometry) — a zone
map is just a ``{node_id: LandUse}`` dict laid over an existing network. The
demand model (:mod:`traffic_sim.demand`) reads it to generate realistic,
time-of-day origin→destination flows (home→work in the morning, etc.), and
regional speed limits can be derived from it.

The default layout is a compact city: a central **office** district (the CBD),
**residential** suburbs around it, and **retail** (shops) scattered throughout.
"""

import random
from enum import Enum
from typing import Dict, List

# A zone map: node id -> land use.
ZoneMap = Dict[int, "LandUse"]


class LandUse(Enum):
    """What a place is used for — the driver of where trips start and end."""

    RESIDENTIAL = "residential"
    OFFICE = "office"
    RETAIL = "retail"
    OTHER = "other"


def assign_zones(net, *, seed: int = 0, retail_fraction: float = 0.12,
                 office_radius: float = 0.42) -> ZoneMap:
    """Tag every node in ``net`` with a :class:`LandUse` and return the map.

    Layout: nodes within ``office_radius`` (as a fraction of the grid's
    half-diagonal) of the centre are the **office** CBD; the rest are
    **residential** suburbs; then a random ``retail_fraction`` of nodes are
    overridden to **retail** shops, scattered everywhere. Seeded, so the same
    network + arguments always give the same zoning.
    """
    rng = random.Random(seed)
    width = max(n.i for n in net.nodes) + 1
    height = max(n.j for n in net.nodes) + 1
    ci, cj = (width - 1) / 2.0, (height - 1) / 2.0
    half_diag = (ci * ci + cj * cj) ** 0.5 or 1.0

    zones: ZoneMap = {}
    for n in net.nodes:
        if n.level != 0:
            continue  # you don't park on an elevated highway
        if rng.random() < retail_fraction:
            zones[n.id] = LandUse.RETAIL
            continue
        dist = ((n.i - ci) ** 2 + (n.j - cj) ** 2) ** 0.5 / half_diag
        zones[n.id] = LandUse.OFFICE if dist <= office_radius else LandUse.RESIDENTIAL
    return zones


def nodes_by_zone(zones: ZoneMap) -> Dict["LandUse", List[int]]:
    """Invert a zone map into ``{land use: [node ids]}`` (only non-empty zones)."""
    out: Dict[LandUse, List[int]] = {}
    for node_id, use in zones.items():
        out.setdefault(use, []).append(node_id)
    return out


# Slow speed for residential streets: 30 km/h in m/s.
RESIDENTIAL_SPEED = 30.0 * 1000.0 / 3600.0


def apply_zone_speeds(net, zones: ZoneMap, *,
                      residential_speed: float = RESIDENTIAL_SPEED) -> None:
    """Slow **local** streets that lie inside the residential zone, in place.

    An edge counts as residential-local when it is a single-lane street (arterials
    and the ring keep their speed — they are through-roads, not neighbourhood
    streets) with *both* endpoints in :attr:`LandUse.RESIDENTIAL`. This models
    traffic-calmed suburbs (30 km/h) and gives the router a reason to prefer the
    faster arterials/ring for through trips. Mutates ``net`` — call it before
    building a router (its cost tables read the speed limits).
    """
    for e in net.edges:
        if (e.lanes == 1
                and zones.get(e.u) is LandUse.RESIDENTIAL
                and zones.get(e.v) is LandUse.RESIDENTIAL):
            e.speed_limit = residential_speed
