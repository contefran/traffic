"""Error analysis of the flow model: what it relies on, and where it misses.

Two questions, both answered on held-out test days with the promoted
scikit-learn trees:

* **Which inputs does the model actually use?** Permutation importance
  (``sklearn.inspection``): shuffle one feature column at a time and measure
  how much the error grows. A feature whose shuffling changes nothing was
  decoration; a feature whose shuffling wrecks the forecast is load-bearing.
* **Where are the misses?** The per-cell absolute errors, sliced by time of
  day, by street class, and by individual street (the worst offenders, with
  their map positions) — the "read your failures" loop from the ML manual:
  every error cluster is either a missing feature, a data artefact, or
  genuine unpredictability, and each deserves a name.

Run (after generating a dataset)::

    MPLBACKEND=Agg bin/python -m ml.analysis --data ml/data/varied
"""

import argparse

import numpy as np
from sklearn.inspection import permutation_importance

from ml.baselines import load_runs, split_runs
from ml.sklearn_models import SKTreesFlowModel


def street_classes(run):
    """``{class name: boolean edge mask}`` from the stored static features."""
    lim, lvl = run["edge_speed_limit"], run["edge_level"]
    return {
        "residential": (lvl == 0) & (lim < 9.0),
        "local": (lvl == 0) & (lim >= 9.0) & (lim < 15.0),
        "arterial": (lvl == 0) & (lim >= 15.0),
        "elevated": lvl == 1,
    }


def analyze(data_dir, *, target="counts", horizon_s=60.0, test_seeds=(5,),
            val_seeds=(4,), n_repeats=3, **knobs):
    """Fit, then dissect the test-day errors. Returns a result dict.

    Keys: ``importances`` (name -> mean MAE increase when shuffled, sorted),
    ``hourly_mae`` (24 bins over the day), ``class_mae`` (street class ->
    MAE), ``worst_edges`` (top 20 edge ids by MAE with their values), plus
    the fitted ``model`` and the units implied by ``target``.
    """
    runs, manifest = load_runs(data_dir)
    rest, test = split_runs(runs, test_seeds)
    train, val = split_runs(rest, val_seeds)
    h = max(1, round(horizon_s / manifest["bin_s"]))
    model = SKTreesFlowModel(h, manifest["day_length"], target=target,
                             **knobs).fit(train, val)

    # --- permutation importance on the pooled test matrix -------------------
    X, y = model._matrix(test)
    imp = permutation_importance(model._model, X, y, n_repeats=n_repeats,
                                 random_state=0,
                                 scoring="neg_mean_absolute_error")
    names = model.feature_names()
    order = np.argsort(imp.importances_mean)[::-1]
    importances = [(names[j], float(imp.importances_mean[j])) for j in order]

    # --- error slices --------------------------------------------------------
    day_len = manifest["day_length"]
    hour_err, hour_n = np.zeros(24), np.zeros(24)
    class_err = {c: [0.0, 0] for c in street_classes(test[0])}
    edge_err = np.zeros(test[0]["counts"].shape[1])
    edge_n = np.zeros_like(edge_err)
    for day in test:
        truth = day[target]
        err = np.abs(model.predict_day(day) - truth)
        scored = np.isfinite(truth)
        scored[:h] = False
        err = np.where(scored, err, 0.0)
        hours = np.minimum((day["bin_t"] / day_len * 24).astype(int), 23)
        for hr in range(24):
            m = hours == hr
            hour_err[hr] += err[m].sum()
            hour_n[hr] += scored[m].sum()
        for cname, cmask in street_classes(day).items():
            class_err[cname][0] += err[:, cmask].sum()
            class_err[cname][1] += scored[:, cmask].sum()
        edge_err += err.sum(axis=0)
        edge_n += scored.sum(axis=0)

    hourly = np.where(hour_n > 0, hour_err / np.maximum(hour_n, 1), np.nan)
    per_class = {c: v[0] / v[1] for c, v in class_err.items() if v[1]}
    per_edge = np.where(edge_n > 0, edge_err / np.maximum(edge_n, 1), 0.0)
    worst = np.argsort(per_edge)[::-1][:20]
    return {
        "target": target, "horizon_s": h * manifest["bin_s"],
        "importances": importances, "hourly_mae": hourly,
        "class_mae": per_class,
        "worst_edges": [(int(e), float(per_edge[e])) for e in worst],
        "per_edge_mae": per_edge, "model": model,
    }


def main(argv=None):
    """CLI: print the dissection for one target/horizon."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/varied")
    p.add_argument("--target", choices=("speed", "counts"), default="counts")
    p.add_argument("--horizon", type=float, default=60.0)
    args = p.parse_args(argv)
    res = analyze(args.data, target=args.target, horizon_s=args.horizon)
    unit = "km/h" if args.target == "speed" else "cars/street"
    scale = 3.6 if args.target == "speed" else 1.0

    print(f"target={args.target}, horizon {res['horizon_s']:.0f} s "
          f"({res['model'].n_trees_} trees)")
    print("top features (MAE increase when shuffled):")
    for name, delta in res["importances"][:8]:
        print(f"  {name:<28} {delta * scale:8.4f} {unit}")
    print("MAE by street class:")
    for c, v in sorted(res["class_mae"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<12} {v * scale:8.4f} {unit}")
    print("worst 5 streets (edge id, MAE):")
    for e, v in res["worst_edges"][:5]:
        print(f"  edge {e:<6} {v * scale:8.4f} {unit}")


if __name__ == "__main__":
    main()
