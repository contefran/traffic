"""The baselines a learned signal timing must beat, measured on the protocol.

Runs the two no-learning baselines of ``guides_plans/rl_plan.md`` §5 over the
common-random-numbers seed protocol (train {1-5} / validation {6, 7} / test
{8-12}) and scores the green wave against the default plans seed-by-seed:

* **Default plans** — every node on the uniform default ``SignalPlan``. This
  *defines* the zero of the J scale (each day scores exactly ``1 - w``
  against itself).
* **Green wave** (``apply_green_wave``, encoded through the same
  :class:`ParameterSpace` vector path the optimizer will use) — the classical
  coordination baseline, i.e. the honest floor.

Usage::

    MPLBACKEND=Agg bin/python -m ml.opt.baselines [--workers N]

Prints the per-seed table and per-split means, and saves the figure to
``experiments/figures/opt_baselines.png``.
"""

import argparse
from typing import Optional, Sequence

from ml.opt.env import (DEFAULT_W, evaluate, green_wave_vector,
                        measure_baselines)

TRAIN_SEEDS = (1, 2, 3, 4, 5)
VAL_SEEDS = (6, 7)
TEST_SEEDS = (8, 9, 10, 11, 12)
ALL_SEEDS = TRAIN_SEEDS + VAL_SEEDS + TEST_SEEDS


def split_of(seed: int) -> str:
    """Which protocol split a seed belongs to (``?`` for ad-hoc seeds)."""
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in VAL_SEEDS:
        return "val"
    return "test" if seed in TEST_SEEDS else "?"


def run(seeds: Sequence[int] = ALL_SEEDS, w: float = DEFAULT_W,
        workers: Optional[int] = None, **overrides) -> list:
    """Measure default + green-wave days on ``seeds``; one result row per seed.

    ``overrides`` pass through to the city build (tests use a small config).
    """
    base = measure_baselines(seeds, workers=workers, **overrides)
    gw = evaluate(green_wave_vector(**overrides), seeds, base,
                  w=w, workers=workers, **overrides)
    rows = []
    for s, day, score in zip(gw.seeds, gw.days, gw.scores):
        b = base[s]
        rows.append({
            "seed": s, "split": split_of(s),
            "trips_base": b["trips_completed"],
            "trips_gw": day["trips_completed"],
            "delay_base": b["mean_delay_s"], "delay_gw": day["mean_delay_s"],
            "crashes_base": b["crashes"], "crashes_gw": day["crashes"],
            "J_gw": score,
        })
    return rows


def print_table(rows: list, w: float = DEFAULT_W) -> None:
    """The per-seed table and per-split means, J relative to the 1-w zero."""
    print(f"{'seed':>4} {'split':>5} {'trips':>11} {'delay [s]':>13} "
          f"{'crashes':>7} {'J-(1-w)':>8}")
    for r in rows:
        gain = (None if r["J_gw"] is None else r["J_gw"] - (1 - w))
        print(f"{r['seed']:>4} {r['split']:>5} "
              f"{r['trips_base']:>5}->{r['trips_gw']:<5} "
              f"{r['delay_base']:>5.1f}->{r['delay_gw']:<6.1f} "
              f"{r['crashes_base']:>3}->{r['crashes_gw']:<3} "
              + ("rejected" if gain is None else f"{gain:>+8.4f}"))
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part or any(r["J_gw"] is None for r in part):
            continue
        d_trips = sum(r["trips_gw"] - r["trips_base"] for r in part) / len(part)
        d_delay = sum(r["delay_gw"] - r["delay_base"] for r in part) / len(part)
        gain = sum(r["J_gw"] for r in part) / len(part) - (1 - w)
        print(f"{split:>5} mean: {d_trips:+.0f} trips, {d_delay:+.1f} s delay, "
              f"J gain {gain:+.4f}")


def render(rows: list, path: str) -> None:
    """Per-seed green-wave effect vs the default plans, split-annotated."""
    import os

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    seeds = [r["seed"] for r in rows]
    xs = range(len(rows))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Green wave vs default plans, per seeded day",
                 color=ink, fontsize=12)
    panels = [
        (ax1, [r["trips_gw"] - r["trips_base"] for r in rows],
         "Δ trips completed"),
        (ax2, [r["delay_gw"] - r["delay_base"] for r in rows],
         "Δ mean delay [s]"),
    ]
    for ax, vals, label in panels:
        ax.bar(xs, vals, width=0.62, color=blue)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_ylabel(label, color=ink)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color=muted, alpha=0.15)
        # Split boundaries (drawn only where both sides exist).
        splits = [r["split"] for r in rows]
        for i in range(1, len(splits)):
            if splits[i] != splits[i - 1]:
                ax.axvline(i - 0.5, color=muted, lw=0.8, ls="--", alpha=0.6)
    for split in ("train", "val", "test"):
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        if idx:
            ax1.text(sum(idx) / len(idx), ax1.get_ylim()[1], split,
                     ha="center", va="bottom", color=muted, fontsize=9)
    ax2.set_xticks(list(xs), [str(s) for s in seeds])
    ax2.set_xlabel("demand seed", color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--w", type=float, default=DEFAULT_W)
    args = parser.parse_args(argv)
    rows = run(w=args.w, workers=args.workers)
    print_table(rows, w=args.w)
    render(rows, "experiments/figures/opt_baselines.png")
    print("figure: experiments/figures/opt_baselines.png")


if __name__ == "__main__":
    main()
