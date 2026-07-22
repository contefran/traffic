"""Q18 — protected vs permissive left turns.

Two ways to handle left turns at a signal:

* **permissive** (:class:`FixedTimeController`, 2 phases) — a whole axis goes at
  once; left-turners share the green with oncoming through traffic.
* **protected** (:class:`ProtectedPhaseController`, 4 phases) — lefts get their
  own phase and never cross oncoming traffic.

We compare them at a **matched cycle length** (so the difference is the *phase
structure*, not the timing) across a load sweep **and an arterial-speed sweep**,
measuring delay, throughput and crashes, with several seeded replicate worlds
per cell.

A :class:`PermissiveLeftModel` is injected so permissive lefts realistically
**yield to oncoming through traffic** (wait for a gap); it is inert under
protected phasing, where the opposing through is red while a left runs. This is
what makes the comparison fair: permissive should win when opposing/left demand
is light (gaps are plentiful) and lose ground as it rises (lefts wait and block
the lane behind them), while protected pays a fixed capacity cost. The speed
axis asks whether faster oncoming traffic — shorter usable gaps for the same
spacing — moves the crossover.

Run: ``python -m experiments.q18_protected_vs_permissive``
"""

from typing import List, Optional, Sequence

from traffic_sim import (
    build_city_grid, ShortestPathRouter, TrafficSim, SignalSystem,
    FixedTimeController, ProtectedPhaseController, PriorityModel,
    PermissiveLeftModel, MetricsCollector, kmh_to_ms,
)
from main import spawn_cars
from experiments.common import pmap

YELLOW = 1.5
# Matched cycle: 2*(g_perm+Y) == 4*(g_prot+Y).  With g_prot=4.5, Y=1.5 -> g_perm=10.5.
GREEN_PROTECTED = 4.5
GREEN_PERMISSIVE = 2 * GREEN_PROTECTED + YELLOW   # 10.5 -> both cycles = 24 s

# Fixed series colours (identity, never cycled): protected blue, permissive orange.
COLORS = {"protected": "#1f77b4", "permissive": "#ff7f0e"}


def _controller(name: str):
    """Build the signal controller for ``name`` (``"permissive"`` or ``"protected"``),
    each timed so the two run on a matched cycle (see the cycle constants above)."""
    if name == "permissive":
        return FixedTimeController(green_time=GREEN_PERMISSIVE, yellow=YELLOW)
    return ProtectedPhaseController(green_time=GREEN_PROTECTED, yellow=YELLOW)


def _one(load: int, speed_kmh: float, seed: int, name: str, steps: int) -> dict:
    """One (load, arterial speed, world seed, controller) cell."""
    net = build_city_grid(8, 8, 150.0, seed=seed, jitter=0.22,
                          one_way_prob=0.15, drop_prob=0.12,
                          arterial_every=3, arterial_speed=kmh_to_ms(speed_kmh))
    cars = spawn_cars(net, load, seed=seed)
    signals = SignalSystem(net, _controller(name))
    m = MetricsCollector()
    sim = TrafficSim(net, cars, ShortestPathRouter(net, seed=42),
                     signals=signals, priority=PriorityModel(net),
                     left_turn=PermissiveLeftModel(net), metrics=m)
    for _ in range(steps):
        sim.step(0.1)
    s = m.summary()
    return {
        "load": load,
        "speed_kmh": speed_kmh,
        "seed": seed,
        "controller": name,
        "mean_delay_s": s.get("mean_delay_s", float("nan")),
        "throughput_per_s": s["throughput_per_s"],
        # Delay is measured on *completed* trips only, so at saturation it is
        # survivorship-biased (cars stuck in queues never report); the count of
        # completed trips is the honest companion number.
        "trips": s.get("trips_completed", 0),
        "crashes": s["crashes"],
        "mean_speed": s["avg_speed"],
    }


def run(loads: Sequence[int] = (20, 40, 60, 80, 100),
        speeds_kmh: Sequence[float] = (70.0,),
        seeds: Sequence[int] = (1,),
        steps: int = 1000,
        workers: Optional[int] = None) -> List[dict]:
    """Compare both controllers over the (load, speed) grid; one row per
    (load, speed, seed, controller)."""
    jobs = [(n, v, s, name, steps)
            for v in speeds_kmh for n in loads for s in seeds
            for name in ("permissive", "protected")]
    return pmap(_one, jobs, workers)


def render(rows: List[dict], path: str = "experiments/figures/q18_protected_vs_permissive.png"):
    """Delay- and throughput-vs-load curves per arterial speed (mean over seeds)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    loads = sorted({r["load"] for r in rows})
    speeds = sorted({r["speed_kmh"] for r in rows})

    def series(speed, controller, key):
        """Seed-mean ``key`` for one (speed, controller), ordered like ``loads``."""
        out = []
        for load in loads:
            vals = [r[key] for r in rows
                    if (r["speed_kmh"], r["controller"], r["load"]) ==
                       (speed, controller, load)]
            out.append(sum(vals) / len(vals))
        return out

    fig, axes = plt.subplots(3, len(speeds), figsize=(4.2 * len(speeds) + 1.5, 10.0),
                             sharex=True, sharey="row", squeeze=False)
    for col, speed in enumerate(speeds):
        ax_d, ax_t, ax_n = axes[0][col], axes[1][col], axes[2][col]
        for name, color in COLORS.items():
            ax_d.plot(loads, series(speed, name, "mean_delay_s"), "o-",
                      color=color, label=name)
            ax_t.plot(loads, series(speed, name, "throughput_per_s"), "o-",
                      color=color, label=name)
            ax_n.plot(loads, series(speed, name, "trips"), "o-",
                      color=color, label=name)
        ax_d.set_title(f"arterials {speed:.0f} km/h")
        ax_n.set_xlabel("cars")
        for ax in (ax_d, ax_t, ax_n):
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("mean delay [s]\n(completed trips only)")
    axes[1][0].set_ylabel("throughput [veh/s]")
    axes[2][0].set_ylabel("trips completed")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Q18 — protected vs permissive left "
                 "(matched 24 s cycle; permissive lefts yield; mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # Loads reach deep into saturation (the 8x8 grid holds ~600 cars while
    # staying placeable) — the interesting question is where, if anywhere,
    # protected phasing overtakes permissive as the junctions jam.
    rows = run(loads=(25, 50, 100, 200, 400, 600),
               speeds_kmh=(50.0, 70.0, 90.0), seeds=(1, 2, 3), steps=3000)
    print(f"{'load':>4s} {'km/h':>5s} {'seed':>4s} {'controller':>11s} "
          f"{'delay':>7s} {'thru/s':>7s} {'trips':>6s} {'crashes':>7s}")
    for r in rows:
        print(f"{r['load']:4d} {r['speed_kmh']:5.0f} {r['seed']:4d} "
              f"{r['controller']:>11s} {r['mean_delay_s']:7.1f} "
              f"{r['throughput_per_s']:7.2f} {r['trips']:6d} {r['crashes']:7d}")
    print("figure:", render(rows))
