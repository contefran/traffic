"""Q20 — traffic statistics by driver personality (and vehicle class).

Every driver already *has* a personality: :func:`~traffic_sim.vehicles
.apply_vehicle_type` jitters the behavioural gaps (+-10%) around the vehicle
type's base values, so some drivers tailgate and some hang back. This
experiment makes that visible in the statistics. Each car gets a **personality
score** — its following gap (``time_headway`` and ``s0``) relative to its
vehicle type's base, so a cautious truck driver and a cautious sports-car
driver compare fairly — and the fleet is split into terciles (*assertive /
average / cautious*). Completed trips from several seeded default-city days are
then aggregated per personality tercile and, separately, per vehicle class.

Notes for reading the result: delay is measured against the *road's* free-flow
time, so heavy vehicles (capped ``max_speed`` below fast-road limits) carry a
built-in delay — that is the vehicle-class effect, and it is expected to dwarf
the +-10% personality effect. Crashes (~2 per day city-wide) are far too few to
segment by personality; they are reported per day in Q19 instead.

Run: ``python -m experiments.q20_personality``
"""

from collections import defaultdict
from typing import List, Optional, Sequence

from traffic_sim.vehicles import DEFAULT_FLEET
from experiments.common import build_default, pmap

PERSONALITIES = ("assertive", "average", "cautious")
VTYPES = tuple(vt.name for _, vt in DEFAULT_FLEET)

_BASE = {vt.name: vt for _, vt in DEFAULT_FLEET}


def personality_score(car) -> float:
    """The car's following gap relative to its vehicle type's base (1.0 = the
    type's textbook driver; <1 tailgates, >1 hangs back)."""
    base = _BASE[car.vtype]
    return 0.5 * (car.time_headway / base.time_headway + car.s0 / base.s0)


def _label_terciles(cars) -> dict:
    """``{car_id: personality}`` by splitting the score's ranking into thirds."""
    ordered = sorted(cars, key=personality_score)
    n = len(ordered)
    return {c.id: PERSONALITIES[min(2, rank * 3 // n)]
            for rank, c in enumerate(ordered)}


def _one_day(seed: int, overrides: dict) -> list:
    """One default-city day; per completed trip: (vtype, personality, delay,
    stops, stopped_time, travel_time)."""
    net, sim, zones, args = build_default(car_seed=seed, **overrides)
    label = _label_terciles(sim.cars)
    vtype = {c.id: c.vtype for c in sim.cars}
    for _ in range(args.steps):
        sim.step(args.dt)
    return [(vtype[tp.car_id], label[tp.car_id], tp.delay, tp.stops,
             tp.stopped_time, tp.travel_time)
            for tp in sim.metrics.trips]


def run(seeds: Sequence[int] = (1, 2, 3),
        workers: Optional[int] = None, **overrides) -> List[dict]:
    """Pool trips over ``seeds`` days; one aggregate row per group, for both
    groupings (personality terciles and vehicle classes)."""
    samples = [t for day in pmap(_one_day, [(s, overrides) for s in seeds], workers)
               for t in day]
    rows: List[dict] = []
    for grouping, key, order in (("personality", 1, PERSONALITIES),
                                 ("vtype", 0, VTYPES)):
        by = defaultdict(list)
        for t in samples:
            by[t[key]].append(t)
        for group in order:
            trips = by.get(group, [])
            if not trips:
                continue
            n = len(trips)
            delays = sorted(t[2] for t in trips)
            rows.append({
                "grouping": grouping,
                "group": group,
                "trips": n,
                "mean_delay_s": sum(delays) / n,
                "median_delay_s": delays[n // 2],
                "p90_delay_s": delays[min(n - 1, int(0.9 * n))],
                "stops_per_trip": sum(t[3] for t in trips) / n,
                "stopped_s_per_trip": sum(t[4] for t in trips) / n,
                "mean_travel_s": sum(t[5] for t in trips) / n,
            })
    return rows


def render(rows: List[dict], path: str = "experiments/figures/q20_personality.png"):
    """Mean delay per group (p90 as a whisker), for both groupings."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, grouping, title in ((axes[0], "personality", "By driver personality"),
                                (axes[1], "vtype", "By vehicle class")):
        rs = [r for r in rows if r["grouping"] == grouping]
        xs = range(len(rs))
        ax.bar(xs, [r["mean_delay_s"] for r in rs], width=0.55, color="#1f77b4")
        ax.errorbar(xs, [r["mean_delay_s"] for r in rs],
                    yerr=[[0.0] * len(rs),
                          [r["p90_delay_s"] - r["mean_delay_s"] for r in rs]],
                    fmt="none", ecolor="0.3", capsize=4)
        ax.set_xticks(list(xs),
                      [f"{r['group']}\n{r['stops_per_trip']:.1f} stops/trip"
                       for r in rs])
        ax.set_ylabel("mean trip delay [s]  (whisker = p90)")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Q20 — trip statistics by driver personality and vehicle class "
                 "(pooled seeded days, full default city)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    rows = run()
    print(f"{'grouping':>11s} {'group':>10s} {'trips':>6s} {'delay':>7s} "
          f"{'median':>7s} {'p90':>7s} {'stops':>6s} {'stop_s':>7s} {'travel':>7s}")
    for r in rows:
        print(f"{r['grouping']:>11s} {r['group']:>10s} {r['trips']:6d} "
              f"{r['mean_delay_s']:7.1f} {r['median_delay_s']:7.1f} "
              f"{r['p90_delay_s']:7.1f} {r['stops_per_trip']:6.2f} "
              f"{r['stopped_s_per_trip']:7.1f} {r['mean_travel_s']:7.1f}")
    print("figure:", render(rows))
