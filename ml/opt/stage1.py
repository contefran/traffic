"""Stage 1: structured low-dimensional search over signal timing.

Instead of 1670 free numbers, four **global knobs** that span the classical
solutions (``guides_plans/rl_plan.md`` §6), each expanded into a full
per-node vector by :func:`structured_vector`:

* ``cycle`` — scales every green time (cycle length: longer = more capacity
  per switch, more waiting per red).
* ``through`` — share of each orientation's green budget given to its
  through phase vs its left phase (0.5 = the default equal split).
* ``bias`` — extra green ratio for the *faster* orientation at nodes where
  an arterial crosses a local street (1 = no bias). Webster's logic, one
  knob.
* ``wave`` — green-wave progression-speed factor: offsets are the classical
  ``apply_green_wave`` offsets divided by this (1 = the classical wave,
  0 = no offsets at all). Uses the *raw* distance/speed offsets returned by
  ``apply_green_wave`` — the installed plans wrap them per cycle, which
  would not rescale correctly.

The full factorial grid (480 combos × 5 training seeds = 2400 simulated
days, ~1 h on 20 cores) doubles as a **measured response surface** per knob,
and the classical combo ``(1.0, 0.5, 1.0, 1.0)`` sits inside the grid, so
the green-wave baseline is reproduced as an internal consistency check.

Usage::

    MPLBACKEND=Agg bin/python -m ml.opt.stage1 [--workers N]

Writes rows to ``ml/data/opt/stage1_rows.json``, the winner (validated on
the held-out validation seeds) to ``ml/data/opt/stage1_best.json``, and the
response-surface figure to ``experiments/figures/opt_stage1.png``.
"""

import argparse
import json
import os
from functools import partial
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.common import build_default, pmap
from ml.opt.baselines import TRAIN_SEEDS, VAL_SEEDS
from ml.opt.env import (DEFAULT_CRASH_CAP, DEFAULT_W, day_score,
                        measure_baselines, run_day, wrap_offsets)
from traffic_sim.signals import Orientation, apply_green_wave, edge_orientation
from traffic_sim.tuning import ParameterSpace

GRID = {
    "cycle": (0.6, 0.8, 1.0, 1.4, 1.8, 2.2),
    "through": (0.4, 0.5, 0.6, 0.7),
    "bias": (1.0, 1.3, 1.6, 2.0),
    "wave": (0.0, 0.7, 1.0, 1.5, 2.5),
}
KNOBS = tuple(GRID)  # insertion order: cycle, through, bias, wave
CLASSICAL = {"cycle": 1.0, "through": 0.5, "bias": 1.0, "wave": 1.0}


# ----------------------------------------------------------------- context

def stage1_context(**overrides):
    """Build once what every candidate shares: the space, the raw green-wave
    offsets, and each node's fastest approach speed per orientation."""
    net, sim, zones, args = build_default(**overrides)
    space = ParameterSpace(sim.signals)
    base_offsets = apply_green_wave(sim.signals)  # raw {node: seconds}
    node_fast: Dict[int, Dict[Orientation, float]] = {}
    for node in space.nodes:
        fast = {Orientation.HORIZONTAL: 0.0, Orientation.VERTICAL: 0.0}
        for eid in net.nodes[node].in_edges:
            o = edge_orientation(net, eid)
            fast[o] = max(fast[o], net.edges[eid].speed_limit)
        node_fast[node] = fast
    return space, base_offsets, node_fast


def structured_vector(space: ParameterSpace, base_offsets: Dict[int, float],
                      node_fast: Dict[int, Dict[Orientation, float]],
                      cycle: float = 1.0, through: float = 0.5,
                      bias: float = 1.0, wave: float = 1.0) -> List[float]:
    """Expand the four knobs into a full per-node timing vector."""
    budget = sum(space.controller.default_plan.green_times) * cycle
    vec: List[float] = []
    for node in space.nodes:
        fast = node_fast[node]
        v_h, v_v = fast[Orientation.HORIZONTAL], fast[Orientation.VERTICAL]
        ratio = bias if v_h > v_v else (1.0 / bias if v_v > v_h else 1.0)
        w_h = ratio / (1.0 + ratio)  # H's share of the green budget
        for share in (w_h, 1.0 - w_h):  # H phases, then V phases
            vec += [budget * share * through, budget * share * (1.0 - through)]
        vec.append(base_offsets.get(node, 0.0) / wave if wave > 0 else 0.0)
    vec += [1.0] * len(space.group_names)  # untouched speed groups, if any
    return wrap_offsets(space, vec)


# ------------------------------------------------------------------ search

def run(grid: Optional[Dict[str, Sequence[float]]] = None,
        seeds: Sequence[int] = TRAIN_SEEDS, w: float = DEFAULT_W,
        crash_cap: int = DEFAULT_CRASH_CAP, workers: Optional[int] = None,
        batch: int = 24, log=print, **overrides) -> List[dict]:
    """Evaluate every knob combo on the training seeds; one row per combo.

    Days are fanned over workers in batches of ``batch`` combos so progress
    is visible; each row carries the knobs, per-seed-mean J (``None`` if any
    day tripped the crash cap) and the trip/delay deltas vs baseline.
    """
    grid = grid or GRID
    space, base_offsets, node_fast = stage1_context(**overrides)
    baselines = measure_baselines(seeds, workers=workers, **overrides)
    combos = [dict(zip(KNOBS, values))
              for values in product(*(grid[k] for k in KNOBS))]
    log(f"stage1: {len(combos)} combos x {len(seeds)} seeds "
        f"= {len(combos) * len(seeds)} days")

    rows: List[dict] = []
    best: Optional[dict] = None
    for start in range(0, len(combos), batch):
        chunk = combos[start:start + batch]
        vectors = [structured_vector(space, base_offsets, node_fast, **c)
                   for c in chunk]
        jobs = [(vec, s) for vec in vectors for s in seeds]
        days = pmap(partial(run_day, **overrides), jobs, workers=workers)
        for i, combo in enumerate(chunk):
            mine = days[i * len(seeds):(i + 1) * len(seeds)]
            scores = [day_score(d, baselines[s], w, crash_cap)
                      for s, d in zip(seeds, mine)]
            row = dict(combo)
            row["J"] = (None if any(sc is None for sc in scores)
                        else sum(scores) / len(scores))
            row["d_trips"] = (sum(d["trips_completed"] for d in mine)
                              - sum(b["trips_completed"]
                                    for b in baselines.values())) / len(seeds)
            row["d_delay"] = (sum(d["mean_delay_s"] for d in mine)
                              - sum(b["mean_delay_s"]
                                    for b in baselines.values())) / len(seeds)
            row["max_crashes"] = max(d["crashes"] for d in mine)
            rows.append(row)
            if row["J"] is not None and (best is None or row["J"] > best["J"]):
                best = row
        done = min(start + batch, len(combos))
        log(f"stage1: {done}/{len(combos)} combos, best so far "
            f"J-(1-w)={best['J'] - (1 - w):+.4f} at "
            + ", ".join(f"{k}={best[k]}" for k in KNOBS))
    return rows


# ---------------------------------------------------------------- reporting

def top_rows(rows: List[dict], n: int = 10) -> List[dict]:
    """The ``n`` best-scoring combos (rejected ones excluded)."""
    return sorted((r for r in rows if r["J"] is not None),
                  key=lambda r: r["J"], reverse=True)[:n]


def render(rows: List[dict], path: str, w: float = DEFAULT_W) -> None:
    """Response surface: per knob value, the best J achievable (profile max)."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    fig.suptitle("Stage 1 response surface: best J gain per knob value "
                 "(profile max over the other knobs)", color=ink, fontsize=11)
    scored = [r for r in rows if r["J"] is not None]
    for ax, knob in zip(axes.flat, KNOBS):
        values = sorted({r[knob] for r in rows})
        peaks = []
        for v in values:
            sub = [r["J"] - (1 - w) for r in scored if r[knob] == v]
            peaks.append(max(sub) if sub else float("nan"))
        ax.plot(range(len(values)), peaks, "-o", color=blue, ms=5)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_xticks(range(len(values)), [str(v) for v in values])
        ax.set_xlabel(knob, color=ink)
        ax.set_ylabel("best J − (1−w)", color=ink)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color=muted, alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--w", type=float, default=DEFAULT_W)
    args = parser.parse_args(argv)

    rows = run(w=args.w, workers=args.workers)
    os.makedirs("ml/data/opt", exist_ok=True)
    with open("ml/data/opt/stage1_rows.json", "w") as fh:
        json.dump(rows, fh, indent=1)

    print(f"\n{'cycle':>5} {'through':>7} {'bias':>5} {'wave':>5} "
          f"{'J-(1-w)':>8} {'d_trips':>8} {'d_delay':>8} {'crash':>5}")
    for r in top_rows(rows):
        print(f"{r['cycle']:>5} {r['through']:>7} {r['bias']:>5} {r['wave']:>5} "
              f"{r['J'] - (1 - args.w):>+8.4f} {r['d_trips']:>+8.1f} "
              f"{r['d_delay']:>+8.2f} {r['max_crashes']:>5}")

    classical = next(r for r in rows
                     if all(r[k] == CLASSICAL[k] for k in KNOBS))
    print(f"\nclassical green wave in-grid: "
          f"J-(1-w)={classical['J'] - (1 - args.w):+.4f} "
          f"(baselines.py measured +0.0635 on train — should match)")

    # Validate the winner on the held-out validation seeds.
    best = top_rows(rows, 1)[0]
    space, base_offsets, node_fast = stage1_context()
    vec = structured_vector(space, base_offsets, node_fast,
                            **{k: best[k] for k in KNOBS})
    val_base = measure_baselines(VAL_SEEDS, workers=args.workers)
    from ml.opt.env import evaluate
    val = evaluate(vec, VAL_SEEDS, val_base, w=args.w, workers=args.workers)
    print(f"winner on validation seeds: J-(1-w)="
          + ("rejected" if val.rejected else f"{val.J - (1 - args.w):+.4f}"))

    with open("ml/data/opt/stage1_best.json", "w") as fh:
        json.dump({"knobs": {k: best[k] for k in KNOBS},
                   "train_J_gain": best["J"] - (1 - args.w),
                   "val_J_gain": None if val.rejected
                   else val.J - (1 - args.w),
                   "w": args.w}, fh, indent=1)
    render(rows, "experiments/figures/opt_stage1.png", w=args.w)
    print("figure: experiments/figures/opt_stage1.png")


if __name__ == "__main__":
    main()
