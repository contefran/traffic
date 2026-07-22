"""Q19 — are bigger following gaps the way to fewer crashes? (No.)

A default-city day ends with ~2 genuine collisions, which invites an obvious
"fix": make every driver keep more distance. This experiment is the presentable
version of why we don't do that. It sweeps a fleet-wide multiplier on every
driver's following gap — ``time_headway`` *and* standstill ``s0`` together,
exactly the dashboard's "following gap x" knob — over several seeded days of the
**full default city** (1000-car mixed fleet, schedules, elevated ring), and
measures crashes against the traffic cost.

The picture: crashes barely respond — ~1 per day at every setting from 0.7x to
1.35x (shrinking every gap 30% adds none, which is the tell that gap size is
not the crash mechanism), and even *doubling* every gap only drifts the seed
mean from ~0.8 to ~0.2 because the residual collisions are stochastic one-off
events (a high-speed queue-back nudge on the expressway, a sub-``dt`` spillback
touch at a congested signal). Meanwhile the traffic cost explodes: mean delay
roughly doubles and the number of trips the city completes in a day collapses
by ~85%. Distance is therefore an analysis knob, not a safety lever; the crash-
elimination work that *did* pay off was structural (yield point-of-no-return
guards, the transfer entry-speed cap, speed-aware unparking — see
``simulation.py``).

Run: ``python -m experiments.q19_following_gap``
"""

from typing import List, Optional, Sequence

from experiments.common import build_default, pmap


def _one_day(scale: float, seed: int, overrides: dict) -> dict:
    """One full default-city day with every following gap scaled by ``scale``."""
    net, sim, zones, args = build_default(car_seed=seed, **overrides)
    for c in sim.cars:
        c.time_headway *= scale
        c.s0 *= scale
    for _ in range(args.steps):
        sim.step(args.dt)
    s = sim.metrics.summary()
    return {
        "gap_scale": scale,
        "seed": seed,
        "crashes": s["crashes"],
        "mean_delay_s": s.get("mean_delay_s", float("nan")),
        "stops_per_trip": s.get("mean_stops_per_trip", float("nan")),
        "mean_speed": s["avg_speed"],
        "fuel_proxy": s["fuel_proxy"],
        "trips": s.get("trips_completed", 0),
    }


def run(gap_scales: Sequence[float] = (0.7, 1.0, 1.35, 1.7, 2.0),
        seeds: Sequence[int] = (1, 2, 3, 4, 5),
        workers: Optional[int] = None, **overrides) -> List[dict]:
    """One row per (gap scale, seeded day); ``overrides`` shrink the city for tests."""
    jobs = [(sc, sd, overrides) for sc in gap_scales for sd in seeds]
    return pmap(_one_day, jobs, workers)


def render(rows: List[dict], path: str = "experiments/figures/q19_following_gap.png"):
    """Crashes (flat, stochastic) vs the traffic cost (rising) across the sweep."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    scales = sorted({r["gap_scale"] for r in rows})

    def per_scale(key):
        return [[r[key] for r in rows if r["gap_scale"] == s] for s in scales]

    def means(key):
        return [sum(v) / len(v) for v in per_scale(key)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Crashes: every seeded day as a dot, the mean as a line. The interesting
    # feature is the *absence* of a downward trend.
    for s, days in zip(scales, per_scale("crashes")):
        ax1.plot([s] * len(days), days, "o", color="0.55", alpha=0.6, ms=5)
    ax1.plot(scales, means("crashes"), "-", color="0.2", lw=2, label="mean of seeds")
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("crashes per day")
    ax1.set_title("Safety: barely responds")
    ax1.legend(frameon=False)

    # Delay: mean line with the min–max band over seeds.
    lo = [min(v) for v in per_scale("mean_delay_s")]
    hi = [max(v) for v in per_scale("mean_delay_s")]
    ax2.fill_between(scales, lo, hi, color="#1f77b4", alpha=0.18, lw=0)
    ax2.plot(scales, means("mean_delay_s"), "o-", color="#1f77b4", lw=2)
    ax2.set_ylabel("mean trip delay [s]")
    ax2.set_title("Cost: delay rises")

    ax3.plot(scales, means("trips"), "o-", color="0.35", lw=2)
    ax3.set_ylim(bottom=0)
    ax3.set_ylabel("trips completed per day")
    ax3.set_title("Cost: the city seizes up")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("following-gap multiplier")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Q19 — bigger following gaps don't buy safety, only delay "
                 "(full default city, one dot per seeded day)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    rows = run()
    print(f"{'gap x':>6s} {'seed':>4s} {'crashes':>7s} {'delay':>7s} "
          f"{'stops':>6s} {'fuel':>9s} {'trips':>6s}")
    for r in rows:
        print(f"{r['gap_scale']:6.2f} {r['seed']:4d} {r['crashes']:7d} "
              f"{r['mean_delay_s']:7.1f} {r['stops_per_trip']:6.2f} "
              f"{r['fuel_proxy']:9.0f} {r['trips']:6d}")
    print("figure:", render(rows))
