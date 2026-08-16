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
  measured scorecard.

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


def geometry_statics(net) -> dict:
    """Per-node world coordinates, keyed like the other static arrays.

    With ``edge_u``/``edge_v`` already in the bundle, node coordinates are
    all a frontend needs to draw every street as a segment — the network's
    edges are straight lines between their endpoint nodes by construction
    (:meth:`RoadNetwork.point_on_edge` interpolates linearly).
    """
    return {
        "node_x": np.array([n.x for n in net.nodes], np.float32),
        "node_y": np.array([n.y for n in net.nodes], np.float32),
        "node_level": np.array([n.level for n in net.nodes], np.int16),
    }


def attach_geometry(bundle: dict, net=None) -> dict:
    """Attach street-map geometry to a bundle (in place; returns it).

    The dataset contract fixes the network across runs, so the city can be
    rebuilt deterministically (``meta["overrides"]`` records any non-default
    builder flags; older bundles imply the default city) and its coordinates
    attached after the fact — no refit, no training data needed. The rebuild
    is **validated** against the stored statics (edge count, lengths,
    topology) so a wrong city cannot be attached silently.
    """
    if net is None:
        from experiments.common import build_default   # offline-only import
        net, _, _, _ = build_default(cars=1,
                                     **bundle["meta"].get("overrides", {}))
    statics = bundle["statics"]
    if len(net.edges) != len(statics["edge_length"]):
        raise ValueError(
            f"network mismatch: rebuilt city has {len(net.edges)} edges, "
            f"bundle has {len(statics['edge_length'])}")
    if not (np.array_equal([e.u for e in net.edges], statics["edge_u"])
            and np.array_equal([e.v for e in net.edges], statics["edge_v"])):
        raise ValueError("network mismatch: edge topology differs")
    if not np.allclose([e.length for e in net.edges], statics["edge_length"],
                       rtol=1e-5):
        raise ValueError("network mismatch: edge lengths differ")
    statics.update(geometry_statics(net))
    if "zone_codes" not in bundle["meta"]:
        from ml.dataset import ZONE_CODES              # offline-only import
        bundle["meta"]["zone_codes"] = {u.name: k
                                        for u, k in ZONE_CODES.items()}
    return bundle


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
    bundle = {
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
            "overrides": manifest.get("overrides", {}),
            "zone_codes": manifest.get("zone_codes", {}),
        },
    }
    return attach_geometry(bundle)


def save(bundle: dict, path) -> None:
    """Write the bundle to ``path`` (parent directories created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load(path) -> dict:
    """Read a bundle back; the models predict immediately (no refit)."""
    return joblib.load(path)


def main(argv=None):
    """CLI: fit the default cells and save the bundle — or, with
    ``--attach-geometry``, augment an existing bundle in place (no refit)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/varied")
    p.add_argument("--out", default="ml/models/varied.joblib")
    p.add_argument("--test-seeds", type=int, nargs="+", default=[5])
    p.add_argument("--val-seeds", type=int, nargs="+", default=[4])
    p.add_argument("--attach-geometry", metavar="BUNDLE", default=None,
                   help="attach street-map geometry to this existing bundle "
                        "(validated city rebuild; skips fitting entirely)")
    args = p.parse_args(argv)

    if args.attach_geometry:
        bundle = attach_geometry(load(args.attach_geometry))
        save(bundle, args.attach_geometry)
        print(f"attached geometry ({bundle['meta']['n_edges']} edges) "
              f"to {args.attach_geometry}")
        return

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
