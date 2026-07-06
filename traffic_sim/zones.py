"""Land-use zones: a spatial overlay tagging each node residential / office / retail.

Kept **decoupled** from the network model (``Node`` stays pure geometry) — a zone
map is just a ``{node_id: LandUse}`` dict laid over an existing network. The
demand model (:mod:`traffic_sim.demand`) reads it to generate realistic,
time-of-day origin→destination flows (home→work in the morning, etc.), and
regional speed limits can be derived from it.

The default layout is a **polycentric** city rather than one central CBD:
several scattered **residential**, **office**, and **retail** clusters, *plus* a
per-type sprinkle of each use randomly across the whole map. That "sparse" share
rises from residential → office → retail: a home is almost always in a
neighbourhood, an office often stands alone, and shops appear on nearly every
street. Because demand picks a destination *zone* then a **random node in it**,
scattering the offices spreads the morning commute across the city instead of
funnelling it into a single core (which used to gridlock).
"""

import random
from enum import Enum
from typing import Dict, List, Optional

# A zone map: node id -> land use.
ZoneMap = Dict[int, "LandUse"]


class LandUse(Enum):
    """What a place is used for — the driver of where trips start and end."""

    RESIDENTIAL = "residential"
    OFFICE = "office"
    RETAIL = "retail"
    OTHER = "other"


# How many clusters (distinct districts) of each land use to place. Housing is
# the most clustered — a city is a handful of neighbourhoods; offices form a few
# business districts; retail a couple of shopping centres (the rest of retail is
# scattered, see DEFAULT_SPARSE).
DEFAULT_CLUSTERS: Dict["LandUse", int] = {
    LandUse.RESIDENTIAL: 8,
    LandUse.OFFICE: 3,
    LandUse.RETAIL: 2,
}

# Probability that a node is *randomly* seeded with a given use, scattered
# city-wide, independent of (and overriding) the clusters — so no district is
# ever 100% one use. Ordered residential < office < retail: the "sparse ratio",
# the share of each land use that lives outside its cluster, grows in that order.
DEFAULT_SPARSE: Dict["LandUse", float] = {
    LandUse.RESIDENTIAL: 0.04,
    LandUse.OFFICE: 0.08,
    LandUse.RETAIL: 0.13,
}


def assign_zones(net, *, seed: int = 0,
                 clusters: Optional[Dict["LandUse", int]] = None,
                 sparse: Optional[Dict["LandUse", float]] = None,
                 cluster_radius: float = 0.24,
                 background: "LandUse" = LandUse.OTHER) -> ZoneMap:
    """Tag every ground node in ``net`` with a :class:`LandUse` and return the map.

    Two layers combine (seeded, so identical inputs always zone identically):

    1. **Clusters** — ``clusters`` places that many random district centres per
       land use; a node within ``cluster_radius`` (a fraction of the grid's
       half-diagonal) of the *nearest* centre takes that centre's use.
    2. **Scatter** — with probability ``sparse[use]`` a node is instead seeded
       with ``use`` at random, anywhere. Scatter is resolved **first**, so it
       also peppers clusters (a corner shop among the houses) and no district is
       pure. Retail scatters most, residential least.

    Anything neither clustered nor scattered becomes ``background`` (undeveloped
    :attr:`LandUse.OTHER` by default — never a trip destination). Elevated nodes
    (``level != 0``) are skipped: you don't park on a highway.
    """
    clusters = DEFAULT_CLUSTERS if clusters is None else clusters
    sparse = DEFAULT_SPARSE if sparse is None else sparse
    rng = random.Random(seed)
    width = max(n.i for n in net.nodes) + 1
    height = max(n.j for n in net.nodes) + 1
    half_diag = (((width - 1) / 2.0) ** 2 + ((height - 1) / 2.0) ** 2) ** 0.5 or 1.0
    radius = cluster_radius * half_diag

    # District centres, at random grid positions (drawn before the per-node
    # scatter so the whole assignment stays deterministic under ``seed``).
    centres: List[tuple] = []  # (LandUse, ci, cj)
    for use, count in clusters.items():
        for _ in range(count):
            centres.append((use, rng.uniform(0, width - 1), rng.uniform(0, height - 1)))

    # Resolve scatter high-share -> low-share so the more ubiquitous use wins ties.
    scatter_order = [LandUse.RETAIL, LandUse.OFFICE, LandUse.RESIDENTIAL]

    zones: ZoneMap = {}
    for n in net.nodes:
        if n.level != 0:
            continue  # you don't park on an elevated highway
        # 1. Random scatter first, so even a cluster is never 100% one use.
        r = rng.random()
        acc = 0.0
        chosen: Optional[LandUse] = None
        for use in scatter_order:
            acc += sparse.get(use, 0.0)
            if r < acc:
                chosen = use
                break
        # 2. Otherwise the nearest district centre within its radius.
        if chosen is None:
            best_d = radius
            for use, ci, cj in centres:
                d = ((n.i - ci) ** 2 + (n.j - cj) ** 2) ** 0.5
                if d <= best_d:
                    best_d, chosen = d, use
        # 3. Fallback: undeveloped background (not a trip endpoint).
        zones[n.id] = chosen if chosen is not None else background
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
