"""The two no-learning forecast baselines every flow model must beat.

* **Persistence** — "the speed in ``horizon`` seconds equals the speed now."
  Traffic has enormous inertia, so this is brutally hard to beat; where it
  fails is exactly where prediction is valuable (congestion forming/clearing).
* **Climatology** (historical average) — "the speed on this street at this
  time of day equals its average over the training days at that time." Knows
  the daily rhythm, ignores today's actual situation. Two variants: pooled
  over all training days, and load-matched (averaging only training days with
  the same fleet size — a stronger bar, using one piece of metadata a real
  forecaster wouldn't have).

Conventions shared with the eventual learned model (so comparisons are fair):

* A ``NaN`` speed means *no cars measured*. An empty street is predicted at
  its **speed limit** (arrive on an empty street and you drive at free flow) —
  that is the fallback wherever a baseline has no measurement to lean on.
* Evaluation scores only cells where the **target** is defined (someone
  actually drove there, so there is a truth to be wrong about), and only bins
  ``b >= horizon_bins`` (cells persistence can legally forecast), identically
  for every method.
* Errors are reported in km/h as MAE — overall, vehicle-weighted (busy
  streets matter more), and sliced by regime: **congested** = target below
  half the street's speed limit, else free-flow. Averages hide sins; the
  congested slice is where a model must earn its keep.

Run (after generating a dataset)::

    MPLBACKEND=Agg bin/python -m ml.baselines --data ml/data/default
"""

import argparse
import json
from pathlib import Path

import numpy as np

from traffic_sim.units import ms_to_kmh

CONGESTED_FRAC = 0.5   # target speed below this fraction of the limit = congested


def load_runs(data_dir):
    """Load every run listed in ``data_dir``'s manifest, newest arrays + meta.

    Returns ``(runs, manifest)`` where each run dict carries the ``.npz``
    arrays plus its manifest row (``cars``, ``car_seed``, ...).
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    runs = []
    for row in manifest["runs"]:
        arrays = dict(np.load(data_dir / row["file"]))
        arrays.update(row)
        runs.append(arrays)
    return runs, manifest


def split_runs(runs, test_seeds=(3,)):
    """Split by **whole days**: any run whose ``car_seed`` is in ``test_seeds``
    is a test day, everything else trains. Splitting at the day boundary is
    what makes leakage between near-duplicate neighbouring bins impossible.
    """
    train = [r for r in runs if r["car_seed"] not in set(test_seeds)]
    test = [r for r in runs if r["car_seed"] in set(test_seeds)]
    if not train or not test:
        raise ValueError(f"empty split: {len(train)} train / {len(test)} test days")
    return train, test


def persistence(speed, horizon_bins, fallback):
    """Forecast ``speed[b]`` as ``speed[b - horizon_bins]`` (empty → fallback).

    Returns an array aligned with ``speed``; the first ``horizon_bins`` rows
    are the fallback (nothing to persist from) and are excluded by
    :func:`evaluate_forecast` anyway.
    """
    pred = np.tile(fallback, (speed.shape[0], 1))
    shifted = speed[:-horizon_bins]
    pred[horizon_bins:] = np.where(np.isnan(shifted), pred[horizon_bins:], shifted)
    return pred


def climatology(train_speeds, fallback):
    """Per-(bin, edge) mean speed over the training days, ignoring empty cells.

    ``train_speeds`` is a list of ``[n_bins, n_edges]`` arrays (one per
    training day). Cells that were empty on *every* training day fall back to
    the street's free-flow speed.
    """
    stack = np.stack(train_speeds)
    measured = np.isfinite(stack)
    n = measured.sum(axis=0)
    mean = np.where(measured, stack, 0.0).sum(axis=0) / np.maximum(n, 1)
    return np.where(n > 0, mean, np.tile(fallback, (mean.shape[0], 1)))


def evaluate_forecast(pred, target, counts, speed_limit, horizon_bins):
    """MAE [km/h] of ``pred`` vs ``target``, overall / weighted / by regime.

    Scores only cells with a defined target and ``b >= horizon_bins`` (the
    common forecastable region). The vehicle-weighted MAE weights each cell by
    its mean occupancy; the regime slices split on the target being below
    :data:`CONGESTED_FRAC` of the street's limit.
    """
    valid = np.isfinite(target)
    valid[:horizon_bins] = False
    err = np.abs(pred - target)[valid]
    w = counts[valid]
    congested = (target < CONGESTED_FRAC * speed_limit[None, :])[valid]

    def _mean(x):
        return float(x.mean()) if x.size else float("nan")

    return {
        "mae_kmh": ms_to_kmh(_mean(err)),
        "mae_weighted_kmh": ms_to_kmh((err * w).sum() / w.sum()) if w.sum() else float("nan"),
        "mae_congested_kmh": ms_to_kmh(_mean(err[congested])),
        "mae_freeflow_kmh": ms_to_kmh(_mean(err[~congested])),
        "cells": int(valid.sum()),
        "congested_share": _mean(congested),
    }


def evaluate_occupancy(pred, target, horizon_bins):
    """MAE [cars/street] of an occupancy forecast.

    Counts are defined everywhere (0 = genuinely empty), so unlike
    :func:`evaluate_forecast` there is nothing to mask — only the warm-up
    bins before ``horizon_bins`` are excluded, identically for every method.
    """
    err = np.abs(pred - target)[horizon_bins:]
    return {"mae_cars": float(err.mean()), "cells": int(err.size)}


def run(data_dir, horizon_s=60.0, test_seeds=(3,)):
    """Score both baselines on the held-out days of ``data_dir``'s dataset.

    Returns one result row per (test day, method) — the comparison table any
    learned model's row gets appended to later.
    """
    runs, manifest = load_runs(data_dir)
    train, test = split_runs(runs, test_seeds)
    horizon_bins = max(1, round(horizon_s / manifest["bin_s"]))

    fallback = test[0]["edge_speed_limit"].astype(float)
    clim_all = climatology([r["speed"] for r in train], fallback)

    rows = []
    for day in test:
        same_load = [r["speed"] for r in train if r["cars"] == day["cars"]]
        forecasts = {
            "persistence": persistence(day["speed"], horizon_bins, fallback),
            "climatology": clim_all,
            "climatology (load-matched)": climatology(same_load, fallback),
        }
        for method, pred in forecasts.items():
            row = {"cars": int(day["cars"]), "car_seed": int(day["car_seed"]),
                   "method": method, "horizon_s": horizon_bins * manifest["bin_s"]}
            row.update(evaluate_forecast(pred, day["speed"], day["counts"],
                                         day["edge_speed_limit"], horizon_bins))
            rows.append(row)
    return rows


def main(argv=None):
    """CLI wrapper: print the baseline table for a generated dataset."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/default",
                   help="dataset directory (from ml.dataset)")
    p.add_argument("--horizon", type=float, default=60.0,
                   help="forecast horizon [s]")
    p.add_argument("--test-seeds", type=int, nargs="+", default=[3],
                   help="car seeds whose whole days are held out for testing")
    args = p.parse_args(argv)
    rows = run(args.data, horizon_s=args.horizon, test_seeds=args.test_seeds)

    print(f"MAE [km/h] on held-out days (horizon {args.horizon:.0f} s), "
          f"by fleet size:")
    header = f"{'method':<28}" + "".join(
        f"{cars:>8}" for cars in sorted({r['cars'] for r in rows}))
    print(header)
    for method in ("persistence", "climatology", "climatology (load-matched)"):
        cells = [r for r in rows if r["method"] == method]
        line = f"{method:<28}"
        for cars in sorted({r["cars"] for r in rows}):
            vals = [r["mae_kmh"] for r in cells if r["cars"] == cars]
            line += f"{np.mean(vals):>8.2f}"
        print(line)
    both = [r for r in rows if r["method"] == "persistence"]
    print(f"(congested share of scored cells: "
          f"{np.mean([r['congested_share'] for r in both]):.1%})")


if __name__ == "__main__":
    main()
