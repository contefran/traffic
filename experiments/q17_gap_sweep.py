"""Q17 — can the PriorityModel's gap-acceptance knobs be optimised?

Human drivers accept conservative gaps; autonomous vehicles (fast reaction, V2V)
accept much smaller ones. We sweep a single ``gap_scale`` that shrinks both
``critical_gap`` and ``min_gap`` from human-like (1.0) toward AV-like (0.2) and
measure throughput, delay and crashes at *unsignalized* junctions (``signals=
None``, so the PriorityModel governs every intersection). Smaller gaps should
raise throughput and cut delay — quantifying the self-driving benefit — with
crashes as the guardrail.

Run: ``python -m experiments.q17_gap_sweep``
"""

from typing import List, Sequence

from traffic_sim import (
    build_city_grid, ShortestPathRouter, TrafficSim, PriorityModel,
    MetricsCollector, kmh_to_ms,
)
from main import spawn_cars

# Human-driver reference gaps (PriorityModel defaults); a scale multiplies both.
BASE_CRITICAL_GAP = 2.5   # [s]
BASE_MIN_GAP = 6.0        # [m]


def run(gap_scales: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.3),
        n_cars: int = 80, steps: int = 1000, seed: int = 1) -> List[dict]:
    """Sweep ``gap_scales`` and return a result row per scale."""
    rows: List[dict] = []
    for scale in gap_scales:
        net = build_city_grid(8, 8, 150.0, seed=seed, jitter=0.22,
                              one_way_prob=0.15, drop_prob=0.12,
                              arterial_every=3, arterial_speed=kmh_to_ms(70))
        cars = spawn_cars(net, n_cars, seed=seed)
        priority = PriorityModel(net, critical_gap=BASE_CRITICAL_GAP * scale,
                                 min_gap=BASE_MIN_GAP * scale)
        m = MetricsCollector()
        sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=42),
                         signals=None, priority=priority, metrics=m)
        for _ in range(steps):
            sim.step(0.1)
        s = m.summary()
        rows.append({
            "gap_scale": scale,
            "throughput_per_s": s["throughput_per_s"],
            "mean_delay_s": s.get("mean_delay_s", float("nan")),
            "crashes": s["crashes"],
            "mean_speed": s["avg_speed"],
        })
    return rows


def render(rows: List[dict], path: str = "experiments/figures/q17_gap_sweep.png"):
    """Save a throughput/delay + crashes figure. Returns the path."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = [r["gap_scale"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax1.plot(x, [r["throughput_per_s"] for r in rows], "o-", color="#1f77b4")
    ax1.set_ylabel("throughput [veh/s]")
    ax1.set_title("Throughput")

    ax2.plot(x, [r["mean_delay_s"] for r in rows], "o-", color="#1f77b4")
    ax2.set_ylabel("mean delay [s]")
    ax2.set_title("Delay")

    ax3.plot(x, [r["crashes"] for r in rows], "o-", color="0.3")
    ax3.set_ylabel("crashes")
    ax3.set_ylim(bottom=0)
    ax3.set_title("Crashes (guardrail)")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("gap scale  (1.0 = human,  →0.2 = autonomous)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Q17 — gap acceptance: human vs autonomous (unsignalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    rows = run()
    print(f"{'gap':>5s} {'thru/s':>7s} {'delay':>7s} {'crashes':>7s} {'mean_v':>7s}")
    for r in rows:
        print(f"{r['gap_scale']:5.2f} {r['throughput_per_s']:7.2f} "
              f"{r['mean_delay_s']:7.1f} {r['crashes']:7d} {r['mean_speed']:7.2f}")
    print("figure:", render(rows))
