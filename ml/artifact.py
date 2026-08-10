"""Fit once, save the result: the flow model as a deployable file.

Until now every evaluation refit the model in memory and threw it away —
"the factory without the boxed product" (ML manual, Step 9). This module
boxes it: fit the promoted scikit-learn trees on the training days, measure
them on the held-out test days, and save everything a server needs into one
``joblib`` bundle:

* **The fitted models** — by default the two skill cells (speed at 10 s,
  occupancy at 60 s). Pickling the model object captures the *entire*
  feature pipeline (climatology table, busyness normalizer, cached
  adjacency, the trees themselves), so serving cannot rebuild features
  differently than training did — the training–serving-skew trap closed by
  shipping the pipeline, not a re-implementation of it.
* **The static edge arrays** (limits, lengths, lanes, zones, topology) —
  the network is fixed across runs (the dataset contract), so a client only
  ever sends the *dynamic* observations and the server fills in the rest.
* **Metadata** — where the data came from, bin width, day length, tree
  counts, and each cell's measured test MAE: the bundle carries its own
  honest scorecard.

Run (writes ``ml/models/varied.joblib``, ~a few minutes)::

    MPLBACKEND=Agg bin/python -m ml.artifact --data ml/data/varied
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ml.baselines import (evaluate_forecast, evaluate_occupancy, load_runs,
                          split_runs)
from ml.sklearn_models import SKTreesFlowModel

# The two cells worth serving: each channel at the horizon where it has
# measured skill (speed decays to the rhythm ceiling by 60 s; occupancy is
# where the medium-horizon signal lives).
DEFAULT_CELLS = (("speed", 10.0), ("counts", 60.0))

STATIC_KEYS = ("edge_speed_limit", "edge_length", "edge_lanes", "edge_level",
               "edge_zone", "edge_u", "edge_v")


def cell_key(target: str, horizon_s: float) -> str:
    """The bundle's name for one (target, horizon) model, e.g. ``speed@10``."""
    return f"{target}@{horizon_s:g}"


def fit_bundle(data_dir, cells=DEFAULT_CELLS, *, test_seeds=(5,),
               val_seeds=(4,), **knobs) -> dict:
    """Fit + score every cell and return the bundle dict (not yet saved).

    ``knobs`` go to :class:`~ml.sklearn_models.SKTreesFlowModel` (handy for
    small test fits). The returned dict is exactly what :func:`save` writes
    and :func:`load` reads back.
    """
    runs, manifest = load_runs(data_dir)
    rest, test = split_runs(runs, test_seeds)
    train, val = split_runs(rest, val_seeds)

    models, meta_cells = {}, []
    for target, horizon_s in cells:
        h = max(1, round(horizon_s / manifest["bin_s"]))
        model = SKTreesFlowModel(h, manifest["day_length"], target=target,
                                 **knobs).fit(train, val)
        maes = []
        for day in test:
            pred = model.predict_day(day)
            if target == "speed":
                r = evaluate_forecast(pred, day["speed"], day["counts"],
                                      day["edge_speed_limit"], h)
                maes.append(r["mae_kmh"])
            else:
                r = evaluate_occupancy(pred, day["counts"], h)
                maes.append(r["mae_cars"])
        models[cell_key(target, horizon_s)] = model
        meta_cells.append({
            "target": target, "horizon_s": horizon_s,
            "n_trees": model.n_trees_,
            "test_mae": float(np.mean(maes)),
            "mae_unit": "km/h" if target == "speed" else "cars/street",
        })

    statics = {k: test[0][k] for k in STATIC_KEYS}
    return {
        "models": models,
        "statics": statics,
        "meta": {
            "data_dir": str(data_dir),
            "bin_s": manifest["bin_s"],
            "day_length": manifest["day_length"],
            "n_edges": int(statics["edge_speed_limit"].shape[0]),
            "train_days": len(train), "val_days": len(val),
            "test_days": len(test),
            "cells": meta_cells,
            "sklearn_version": sklearn.__version__,
        },
    }


def save(bundle: dict, path) -> None:
    """Write the bundle to ``path`` (parent directories created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load(path) -> dict:
    """Read a bundle back; the models predict immediately (no refit)."""
    return joblib.load(path)


def main(argv=None):
    """CLI: fit the default cells and save the bundle."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/varied")
    p.add_argument("--out", default="ml/models/varied.joblib")
    p.add_argument("--test-seeds", type=int, nargs="+", default=[5])
    p.add_argument("--val-seeds", type=int, nargs="+", default=[4])
    args = p.parse_args(argv)

    bundle = fit_bundle(args.data, test_seeds=tuple(args.test_seeds),
                        val_seeds=tuple(args.val_seeds))
    save(bundle, args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB)")
    for c in bundle["meta"]["cells"]:
        print(f"  {cell_key(c['target'], c['horizon_s']):<12}"
              f"{c['n_trees']:>4} trees   test MAE "
              f"{c['test_mae']:.4f} {c['mae_unit']}")


if __name__ == "__main__":
    main()
