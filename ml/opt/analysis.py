"""Stage-2 post-run analysis: what the 48 h search did, and the final verdict.

Three questions, answered from the run's saved artifacts plus a handful of
fresh held-out days:

1. **What happened during the search?** The optimization curve — per-
   generation candidate scores from the checkpoint's history, the periodic
   held-out validations, and the warm start's anchors, in one figure. A
   companion panel shows how far the final mean actually drifted from the
   warm start (per knob family, in seconds).
2. **What is the final product worth on untouched seeds?** The delivered
   plan (the best *validated* mean — by construction the stage-1 warm start
   whenever the ES never beat it) scored on the test seeds, next to the
   green-wave baseline, both against the default-timing baseline. These
   seeds were never seen by any search stage, so this is the number that
   counts.
3. **Does the product generalize across load?** The same plan under lighter
   and heavier demand than the 1000 cars it was tuned at (q18's lesson:
   load moves the optimum), against matched same-seed same-load baselines.
   The crash cap is reported, not enforced, here — this is a robustness
   probe, not a training signal.

Usage::

    MPLBACKEND=Agg bin/python -m ml.opt.analysis [--workers N] [--skip-loads]

Writes ``ml/data/opt/stage2_analysis.json`` and two figures under
``experiments/figures/`` (``opt_stage2_curve.png``, ``opt_final.png``).
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from ml.opt.baselines import TEST_SEEDS
from ml.opt.env import (DEFAULT_CRASH_CAP, DEFAULT_W, day_score, evaluate,
                        green_wave_vector, measure_baselines)
from ml.opt.stage1 import stage1_context, structured_vector
from ml.opt.stage2 import CKPT, RESULT, STAGE1_BEST

ANALYSIS = "ml/data/opt/stage2_analysis.json"
CURVE_FIG = "experiments/figures/opt_stage2_curve.png"
FINAL_FIG = "experiments/figures/opt_final.png"
#: Loads for the generalization probe; the tuning load 1000 comes from the
#: test scoreboard (same seeds), so only the off-loads are simulated here.
OFF_LOADS = (500, 1500)
LOAD_SEEDS = (8, 9, 10)


def load_run(checkpoint: str = CKPT, result_path: str = RESULT) -> dict:
    """The saved run: optimizer history + result (with validation history)."""
    with open(checkpoint) as fh:
        ckpt = json.load(fh)
    with open(result_path) as fh:
        result = json.load(fh)
    return {"history": ckpt["history"], "result": result,
            "val_history": result["val_history"]}


def drift_stats(mean: Sequence[float], warm: Sequence[float],
                labels: Sequence[str]) -> dict:
    """Absolute per-dimension |final - warm| deltas, split by knob family."""
    greens, offsets = [], []
    for m, s, lab in zip(mean, warm, labels):
        (offsets if lab.endswith(".offset") else greens).append(abs(m - s))
    mean_of = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"greens": greens, "offsets": offsets,
            "greens_mean_s": mean_of(greens),
            "offsets_mean_s": mean_of(offsets)}


# ---------------------------------------------------------------- scoring

def _row(seed: int, base: dict, day: dict, score: Optional[float],
         w: float) -> dict:
    return {
        "seed": seed,
        "trips": day["trips_completed"], "base_trips": base["trips_completed"],
        "delay_s": day["mean_delay_s"], "base_delay_s": base["mean_delay_s"],
        "crashes": day["crashes"], "base_crashes": base["crashes"],
        "J_gain": None if score is None else score - (1 - w),
    }


def scoreboard(vectors: Dict[str, Sequence[float]],
               seeds: Sequence[int] = TEST_SEEDS, w: float = DEFAULT_W,
               crash_cap: int = DEFAULT_CRASH_CAP,
               workers: Optional[int] = None,
               **overrides) -> Dict[str, List[dict]]:
    """Each named vector scored per seed against the default baseline."""
    base = measure_baselines(seeds, workers=workers, **overrides)
    out = {}
    for name, vec in vectors.items():
        ev = evaluate(vec, seeds, base, w=w, crash_cap=crash_cap,
                      workers=workers, **overrides)
        out[name] = [_row(s, base[s], d, sc, w)
                     for s, d, sc in zip(ev.seeds, ev.days, ev.scores)]
    return out


def load_sweep(vector: Sequence[float], loads: Sequence[int] = OFF_LOADS,
               seeds: Sequence[int] = LOAD_SEEDS, w: float = DEFAULT_W,
               workers: Optional[int] = None, **overrides) -> List[dict]:
    """The product vs matched baselines at off-tuning loads.

    Scores are always computed (no crash-cap rejection); each row carries
    both sides' crash counts so the safety cost is visible rather than
    silently gating the comparison.
    """
    rows = []
    for load in loads:
        base = measure_baselines(seeds, workers=workers, cars=load,
                                 **overrides)
        ev = evaluate(vector, seeds, base, w=w, crash_cap=10 ** 9,
                      workers=workers, cars=load, **overrides)
        for s, d, sc in zip(ev.seeds, ev.days, ev.scores):
            row = _row(s, base[s], d, sc, w)
            row["cars"] = load
            rows.append(row)
    return rows


def mean_gain(rows: List[dict]) -> Optional[float]:
    """Mean J gain over rows (``None`` if any row was rejected)."""
    gains = [r["J_gain"] for r in rows]
    if any(g is None for g in gains):
        return None
    return sum(gains) / len(gains)


# ---------------------------------------------------------------- figures

def render_curve(run: dict, drift: dict, path: str, w: float = DEFAULT_W,
                 anchors: Optional[dict] = None) -> None:
    """The optimization story in one figure: curve left, drift right.

    ``anchors`` carries the warm start's ``train_J_gain``/``val_J_gain``
    reference lines; by default they are read from the stage-1 result file.
    """
    import matplotlib.pyplot as plt

    if anchors is None:
        with open(STAGE1_BEST) as fh:
            anchors = json.load(fh)
    best1 = anchors
    hist = run["history"]
    gens = [h["generation"] for h in hist]
    cand_mean = [None if h["mean_score"] is None else h["mean_score"] - (1 - w)
                 for h in hist]
    cand_best = [None if h["best_in_gen"] is None else h["best_in_gen"] - (1 - w)
                 for h in hist]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                  gridspec_kw={"width_ratios": [3, 2]})
    ax.plot(gens, cand_mean, "o-", color="tab:blue", label="candidate mean")
    ax.plot(gens, cand_best, "s--", color="tab:cyan", alpha=0.8,
            label="best in generation")
    ax.axhline(best1["train_J_gain"], color="tab:green", ls=":",
               label=f"warm start (train) {best1['train_J_gain']:+.3f}")
    ax.axhline(best1["val_J_gain"], color="tab:red", ls=":",
               label=f"warm start (validation) {best1['val_J_gain']:+.3f}")
    vx = [v["generation"] for v in run["val_history"] if v["J"] is not None]
    vy = [v["J"] - (1 - w) for v in run["val_history"] if v["J"] is not None]
    ax.plot(vx, vy, "D-", color="tab:red", ms=8, label="mean on validation")
    floor = min(y for y in cand_mean + vy if y is not None) - 0.01
    for v in run["val_history"]:
        if v["J"] is None:
            ax.plot([v["generation"]], [floor], "x", color="tab:red",
                    ms=10, mew=2)
            ax.annotate("rejected\n(crashes)", (v["generation"], floor),
                        textcoords="offset points", xytext=(0, 6),
                        color="tab:red", fontsize=8, ha="center")
    fv = run["result"].get("final_val_J_gain")
    if fv is not None:
        ax.plot([gens[-1]], [fv], "D", color="darkred", ms=8)
        ax.annotate("final", (gens[-1], fv), textcoords="offset points",
                    xytext=(6, -4), color="darkred", fontsize=8)
    ax.set_xlabel("generation")
    ax.set_ylabel(f"J gain over default plans (w={w})")
    ax.set_title("Stage 2: per-node fine-tune from the stage-1 warm start")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax2.hist(drift["greens"], bins=30, alpha=0.7, color="tab:green",
             label=f"greens (mean {drift['greens_mean_s']:.1f}s)")
    ax2.hist(drift["offsets"], bins=30, alpha=0.6, color="tab:purple",
             label=f"offsets (mean {drift['offsets_mean_s']:.1f}s)")
    ax2.set_xlabel("|final mean - warm start|  [s]")
    ax2.set_ylabel("dimensions")
    ax2.set_title("How far the search actually moved")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_final(board: Dict[str, List[dict]], load_rows: List[dict],
                 path: str, w: float = DEFAULT_W) -> None:
    """The product's verdict: test seeds left, load generalization right."""
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(board)
    colors = {"green wave": "tab:green", "tuned plan": "tab:blue"}
    seeds = [r["seed"] for r in board[names[0]]]
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        gains = [r["J_gain"] for r in board[name]]
        xs = [i + (k - (len(names) - 1) / 2) * width
              for i in range(len(seeds))]
        ax.bar(xs, gains, width=width * 0.9,
               color=colors.get(name, f"C{k}"), label=name)
        mg = mean_gain(board[name])
        if mg is not None:
            ax.axhline(mg, color=colors.get(name, f"C{k}"), ls=":",
                       alpha=0.8)
            ax.annotate(f"mean {mg:+.3f}", (len(seeds) - 0.5, mg),
                        fontsize=8, color=colors.get(name, f"C{k}"),
                        va="bottom", ha="right")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(f"J gain over default plans (w={w})")
    ax.set_title("Held-out test seeds (never searched)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    loads = sorted({r["cars"] for r in load_rows})
    means = [mean_gain([r for r in load_rows if r["cars"] == load])
             for load in loads]
    crash = {load: (sum(r["crashes"] for r in load_rows
                        if r["cars"] == load),
                    sum(r["base_crashes"] for r in load_rows
                        if r["cars"] == load))
             for load in loads}
    ax2.bar([str(l) for l in loads], means, color="tab:blue", width=0.5)
    for i, load in enumerate(loads):
        c, cb = crash[load]
        ax2.annotate(f"crashes {c} (base {cb})", (i, 0.002), ha="center",
                     fontsize=8, color="black")
    ax2.set_xlabel("cars (plan was tuned at 1000)")
    ax2.set_ylabel(f"mean J gain (w={w})")
    ax2.set_title("Load generalization of the tuned plan")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- main

def print_board(board: Dict[str, List[dict]]) -> None:
    for name, rows in board.items():
        print(f"\n{name} vs default baseline:")
        print("  seed  trips(base)      delay(base)     crashes(base)  Jgain")
        for r in rows:
            j = "reject" if r["J_gain"] is None else f"{r['J_gain']:+.4f}"
            print(f"  {r['seed']:>4}  {r['trips']:>5} ({r['base_trips']:>5})"
                  f"  {r['delay_s']:>6.1f}s ({r['base_delay_s']:>6.1f}s)"
                  f"  {r['crashes']:>3} ({r['base_crashes']:>3})       {j}")
        mg = mean_gain(rows)
        print(f"  mean J gain: "
              + ("rejected" if mg is None else f"{mg:+.4f}"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-loads", action="store_true",
                        help="skip the off-load generalization days")
    args = parser.parse_args(argv)

    run = load_run()
    space, base_offsets, node_fast = stage1_context()
    with open(STAGE1_BEST) as fh:
        knobs = json.load(fh)["knobs"]
    warm = structured_vector(space, base_offsets, node_fast, **knobs)
    product = run["result"]["best_val_mean"]
    is_warm = all(abs(a - b) < 1e-9 for a, b in zip(product, warm))
    print(f"product = best validated mean; identical to stage-1 warm start: "
          f"{is_warm}")

    drift = drift_stats(run["result"]["mean"], warm, space.labels())
    print(f"final-mean drift from warm start: greens "
          f"{drift['greens_mean_s']:.2f}s, offsets "
          f"{drift['offsets_mean_s']:.2f}s (mean |delta|)")
    render_curve(run, drift, CURVE_FIG)
    print(f"wrote {CURVE_FIG}")

    print("\nscoring green wave + product on test seeds "
          f"{list(TEST_SEEDS)} ...")
    board = scoreboard({"green wave": green_wave_vector(),
                        "tuned plan": product}, workers=args.workers)
    print_board(board)

    load_rows = []
    if not args.skip_loads:
        print(f"\nload generalization at {list(OFF_LOADS)} cars, seeds "
              f"{list(LOAD_SEEDS)} ...")
        load_rows = load_sweep(product, workers=args.workers)
        for load in OFF_LOADS:
            rows = [r for r in load_rows if r["cars"] == load]
            mg = mean_gain(rows)
            print(f"  {load} cars: mean J gain "
                  + ("rejected" if mg is None else f"{mg:+.4f}")
                  + f", crashes {sum(r['crashes'] for r in rows)} vs base "
                  f"{sum(r['base_crashes'] for r in rows)}")
        board_1000 = [r for r in board["tuned plan"]
                      if r["seed"] in LOAD_SEEDS]
        for r in board_1000:
            row = dict(r)
            row["cars"] = 1000
            load_rows.append(row)
        render_final(board, load_rows, FINAL_FIG)
        print(f"wrote {FINAL_FIG}")

    os.makedirs(os.path.dirname(ANALYSIS) or ".", exist_ok=True)
    with open(ANALYSIS, "w") as fh:
        json.dump({"product_is_warm_start": is_warm,
                   "drift_greens_mean_s": drift["greens_mean_s"],
                   "drift_offsets_mean_s": drift["offsets_mean_s"],
                   "test_board": board, "load_rows": load_rows}, fh,
                  indent=1)
    print(f"wrote {ANALYSIS}")


if __name__ == "__main__":
    main()
