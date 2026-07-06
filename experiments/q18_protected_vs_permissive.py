"""Q18 — protected vs permissive left turns.

Two ways to handle left turns at a signal:

* **permissive** (:class:`FixedTimeController`, 2 phases) — a whole axis goes at
  once; left-turners share the green with oncoming through traffic.
* **protected** (:class:`ProtectedPhaseController`, 4 phases) — lefts get their
  own phase and never cross oncoming traffic.

We compare them at a **matched cycle length** (so the difference is the *phase
structure*, not the timing) across a load sweep, measuring delay, throughput and
crashes.

A :class:`PermissiveLeftModel` is injected so permissive lefts realistically
**yield to oncoming through traffic** (wait for a gap); it is inert under
protected phasing, where the opposing through is red while a left runs. This is
what makes the comparison fair: permissive should win when opposing/left demand
is light (gaps are plentiful) and lose ground as it rises (lefts wait and block
the lane behind them), while protected pays a fixed capacity cost.

Run: ``python -m experiments.q18_protected_vs_permissive``
"""

from typing import List, Sequence

from traffic_sim import (
    build_city_grid, ShortestPathRouter, TrafficSim, SignalSystem,
    FixedTimeController, ProtectedPhaseController, PriorityModel,
    PermissiveLeftModel, MetricsCollector, kmh_to_ms,
)
from main import spawn_cars

YELLOW = 1.5
# Matched cycle: 2*(g_perm+Y) == 4*(g_prot+Y).  With g_prot=4.5, Y=1.5 -> g_perm=10.5.
GREEN_PROTECTED = 4.5
GREEN_PERMISSIVE = 2 * GREEN_PROTECTED + YELLOW   # 10.5 -> both cycles = 24 s


def _controller(name: str):
    """Build the signal controller for ``name`` (``"permissive"`` or ``"protected"``),
    each timed so the two run on a matched cycle (see the cycle constants above)."""
    if name == "permissive":
        return FixedTimeController(green_time=GREEN_PERMISSIVE, yellow=YELLOW)
    return ProtectedPhaseController(green_time=GREEN_PROTECTED, yellow=YELLOW)


def run(loads: Sequence[int] = (20, 40, 60, 80, 100),
        steps: int = 1000, seed: int = 1) -> List[dict]:
    """Compare both controllers across ``loads``; a result row per (load, controller)."""
    rows: List[dict] = []
    for n in loads:
        for name in ("permissive", "protected"):
            net = build_city_grid(8, 8, 150.0, seed=seed, jitter=0.22,
                                  one_way_prob=0.15, drop_prob=0.12,
                                  arterial_every=3, arterial_speed=kmh_to_ms(70))
            cars = spawn_cars(net, n, seed=seed)
            signals = SignalSystem(net, _controller(name))
            m = MetricsCollector()
            sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=42),
                             signals=signals, priority=PriorityModel(net),
                             left_turn=PermissiveLeftModel(net), metrics=m)
            for _ in range(steps):
                sim.step(0.1)
            s = m.summary()
            rows.append({
                "load": n,
                "controller": name,
                "mean_delay_s": s.get("mean_delay_s", float("nan")),
                "throughput_per_s": s["throughput_per_s"],
                "crashes": s["crashes"],
                "mean_speed": s["avg_speed"],
            })
    return rows


def render(rows: List[dict], path: str = "experiments/figures/q18_protected_vs_permissive.png"):
    """Save delay- and throughput-vs-load curves for both controllers."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    loads = sorted({r["load"] for r in rows})

    def series(controller, key):
        """``key`` values for ``controller``, ordered to match ``loads`` (the x-axis)."""
        by = {r["load"]: r[key] for r in rows if r["controller"] == controller}
        return [by[l] for l in loads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, color in (("permissive", "tab:red"), ("protected", "tab:blue")):
        ax1.plot(loads, series(name, "mean_delay_s"), "o-", color=color, label=name)
        ax2.plot(loads, series(name, "throughput_per_s"), "o-", color=color, label=name)
    ax1.set_xlabel("cars"); ax1.set_ylabel("mean delay [s]")
    ax1.set_title("Delay vs load"); ax1.legend()
    ax2.set_xlabel("cars"); ax2.set_ylabel("throughput [veh/s]")
    ax2.set_title("Throughput vs load"); ax2.legend()
    fig.suptitle("Q18 — protected vs permissive left (matched cycle; permissive lefts yield)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    rows = run()
    print(f"{'load':>4s} {'controller':>11s} {'delay':>7s} {'thru/s':>7s} {'crashes':>7s}")
    for r in rows:
        print(f"{r['load']:4d} {r['controller']:>11s} {r['mean_delay_s']:7.1f} "
              f"{r['throughput_per_s']:7.2f} {r['crashes']:7d}")
    print("figure:", render(rows))
