"""Land-use zones: a spatial overlay tagging each **street (edge)** residential /
office / retail.

Kept **decoupled** from the network model (``Edge``/``Node`` stay pure geometry)
— a zone map is just an ``{edge_id: LandUse}`` dict laid over an existing
network. Land use lives on **edges, not intersections**: a house or a shop sits
on a street, not on a junction. The demand model (:mod:`traffic_sim.demand`)
reads it to generate realistic, time-of-day origin→destination flows (home→work
in the morning, etc.), and regional speed limits can be derived from it.

The default layout is a **polycentric** city rather than one central CBD:
several scattered **residential**, **office**, and **retail** clusters, *plus* a
per-type sprinkle of each use randomly across the whole map. That "sparse" share
rises from residential → office → retail: a home is almost always in a
neighbourhood, an office often stands alone, and shops appear on nearly every
street. Because demand picks a destination *zone* then a **random street in it**,
scattering the offices spreads the morning commute across the city instead of
funnelling it into a single core (which used to gridlock).

Both directions of a two-way street share one land use (they are the same
street), and elevated/ramp edges are left unzoned (you don't live on a highway).
"""

import random
from enum import Enum
from typing import Dict, List, Optional

# A zone map: edge id -> land use.
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
    """Tag every ground **edge** in ``net`` with a :class:`LandUse` and return the map.

    A street's land use is decided from its **midpoint** by two layers that
    combine (seeded, so identical inputs always zone identically):

    1. **Clusters** — ``clusters`` places that many random district centres per
       land use; a street whose midpoint is within ``cluster_radius`` (a fraction
       of the grid's half-diagonal) of the *nearest* centre takes that use.
    2. **Scatter** — with probability ``sparse[use]`` a street is instead seeded
       with ``use`` at random, anywhere. Scatter is resolved **first**, so it
       also peppers clusters (a corner shop among the houses) and no district is
       pure. Retail scatters most, residential least.

    A street with neither becomes ``background`` (undeveloped :attr:`LandUse.OTHER`
    by default — never a trip destination). **Both directions of a two-way street
    share one use** (they are the same street). Edges touching an elevated node
    (``level != 0``) are skipped — you don't live on a highway.
    """
    clusters = DEFAULT_CLUSTERS if clusters is None else clusters
    sparse = DEFAULT_SPARSE if sparse is None else sparse
    rng = random.Random(seed)
    width = max(n.i for n in net.nodes) + 1
    height = max(n.j for n in net.nodes) + 1
    half_diag = (((width - 1) / 2.0) ** 2 + ((height - 1) / 2.0) ** 2) ** 0.5 or 1.0
    radius = cluster_radius * half_diag

    # District centres, at random grid positions (drawn before the per-street
    # scatter so the whole assignment stays deterministic under ``seed``).
    centres: List[tuple] = []  # (LandUse, ci, cj)
    for use, count in clusters.items():
        for _ in range(count):
            centres.append((use, rng.uniform(0, width - 1), rng.uniform(0, height - 1)))

    # Resolve scatter high-share -> low-share so the more ubiquitous use wins ties.
    scatter_order = [LandUse.RETAIL, LandUse.OFFICE, LandUse.RESIDENTIAL]

    def classify(mi: float, mj: float) -> "LandUse":
        """Land use for a street midpoint at grid position ``(mi, mj)``."""
        r = rng.random()
        acc = 0.0
        for use in scatter_order:              # 1. random scatter wins first
            acc += sparse.get(use, 0.0)
            if r < acc:
                return use
        best_d, chosen = radius, None          # 2. else nearest district centre
        for use, ci, cj in centres:
            d = ((mi - ci) ** 2 + (mj - cj) ** 2) ** 0.5
            if d <= best_d:
                best_d, chosen = d, use
        return chosen if chosen is not None else background  # 3. undeveloped

    zones: ZoneMap = {}
    pair_use: Dict[tuple, "LandUse"] = {}      # unordered endpoints -> shared use
    for e in net.edges:
        n1, n2 = net.nodes[e.u], net.nodes[e.v]
        if n1.level != 0 or n2.level != 0:
            continue                            # skip elevated / ramp edges
        if n1.internal or n2.internal:
            continue                            # skip roundabout ring / island
        key = (min(e.u, e.v), max(e.u, e.v))
        if key not in pair_use:                 # first direction seen: decide once
            pair_use[key] = classify((n1.i + n2.i) / 2.0, (n1.j + n2.j) / 2.0)
        zones[e.id] = pair_use[key]             # both directions share it
    return zones


def edges_by_zone(zones: ZoneMap) -> Dict["LandUse", List[int]]:
    """Invert a zone map into ``{land use: [edge ids]}`` (only non-empty zones)."""
    out: Dict[LandUse, List[int]] = {}
    for edge_id, use in zones.items():
        out.setdefault(use, []).append(edge_id)
    return out


# Slow speed for residential streets: 30 km/h in m/s.
RESIDENTIAL_SPEED = 30.0 * 1000.0 / 3600.0


def apply_zone_speeds(net, zones: ZoneMap, *,
                      residential_speed: float = RESIDENTIAL_SPEED) -> None:
    """Slow **residential local** streets to 30 km/h, in place.

    An edge is slowed when it is a single-lane street (arterials and the ring keep
    their speed — they are through-roads, not neighbourhood streets) that is itself
    zoned :attr:`LandUse.RESIDENTIAL`. This models traffic-calmed suburbs and gives
    the router a reason to prefer the faster arterials/ring for through trips.
    Mutates ``net`` — call it before building a router (its cost tables read the
    speed limits).
    """
    for e in net.edges:
        if e.lanes == 1 and zones.get(e.id) is LandUse.RESIDENTIAL:
            e.speed_limit = residential_speed
