"""The same two flow models, backed by scikit-learn — the industry library.

The point of this module is the cross-check (and the portfolio line): we can
*build* the models from scratch (:mod:`ml.linear`, :mod:`ml.gbdt`) and we can
*use* the standard external package — and show the two agree. Both classes
subclass :class:`~ml.linear.LinearFlowModel`, so they inherit the exact
feature pipeline and prediction plumbing; only the fitting engine changes:

* :class:`SKRidgeFlowModel` — ``sklearn`` ridge regression (with its scaler),
  replacing our closed-form normal-equation solve.
* :class:`SKTreesFlowModel` — ``HistGradientBoostingRegressor``, scikit-learn's
  histogram-based boosted trees (the same design our from-scratch version
  implements). One care point: its *built-in* early stopping holds out random
  **rows**, which our manual forbids (rows seconds apart are near-duplicates —
  leakage). We keep the whole-days principle by growing the model in steps
  with ``warm_start`` and checking held-out *validation days* ourselves.

A fringe benefit we'll use in the error-analysis step: scikit-learn ships
`sklearn.inspection.permutation_importance` — shuffle one feature at a time
and see how much the score degrades, i.e. which inputs the model actually
relies on. (It is model-agnostic, so it can rank our from-scratch model's
features too.)

Run (after generating a dataset)::

    MPLBACKEND=Agg bin/python -m ml.sklearn_models --data ml/data/varied
"""

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.baselines import (evaluate_forecast, evaluate_occupancy, load_runs,
                          split_runs)
from ml.linear import LinearFlowModel


class SKRidgeFlowModel(LinearFlowModel):
    """Ridge regression via scikit-learn (scaler + solver replace our math)."""

    def fit(self, train_runs) -> "SKRidgeFlowModel":
        """Fit the scaler+ridge pipeline on the training days."""
        self._prepare(train_runs)
        X, y = self._matrix(train_runs)
        self._model = make_pipeline(StandardScaler(),
                                    Ridge(alpha=self.ridge)).fit(X, y)
        return self

    def _apply(self, X) -> np.ndarray:
        return self._model.predict(X)


class SKTreesFlowModel(LinearFlowModel):
    """Boosted trees via scikit-learn, early-stopped on whole validation days."""

    def __init__(self, horizon_bins: int, day_length: float, *,
                 lags: int = 3, target: str = "speed", neighbours: bool = True,
                 max_trees: int = 400, step: int = 25,
                 learning_rate: float = 0.1, max_depth: int = 5,
                 patience_steps: int = 3, seed: int = 0) -> None:
        """``max_trees`` is the ceiling; the model grows ``step`` trees at a
        time and keeps the count that scored best on the validation days
        (stopping after ``patience_steps`` growth steps without improvement).
        """
        super().__init__(horizon_bins, day_length, lags=lags, target=target,
                         neighbours=neighbours)
        self.max_trees = max_trees
        self.step = step
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.patience_steps = patience_steps
        self.seed = seed
        self.n_trees_ = 0

    def _make(self, n_iter: int, warm: bool) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=n_iter, learning_rate=self.learning_rate,
            max_depth=self.max_depth, early_stopping=False,
            warm_start=warm, random_state=self.seed)

    def fit(self, train_runs, val_runs=None) -> "SKTreesFlowModel":
        """Grow on training days; validation days pick the tree count."""
        self._prepare(train_runs)
        X, y = self._matrix(train_runs)

        if val_runs is None:
            self._model = self._make(self.max_trees, warm=False).fit(X, y)
            self.n_trees_ = self.max_trees
            return self

        Xv, yv = self._matrix(val_runs)
        grower = self._make(self.step, warm=True)
        best_err, best_n, since = np.inf, self.step, 0
        for n in range(self.step, self.max_trees + 1, self.step):
            grower.set_params(max_iter=n)
            grower.fit(X, y)
            err = np.abs(grower.predict(Xv) - yv).mean()
            if err < best_err - 1e-9:
                best_err, best_n, since = err, n, 0
            else:
                since += 1
                if since >= self.patience_steps:
                    break
        # Refit clean at the chosen size (a warm-grown model has extra trees).
        self._model = self._make(best_n, warm=False).fit(X, y)
        self.n_trees_ = best_n
        return self

    def _apply(self, X) -> np.ndarray:
        return self._model.predict(X)


def run(data_dir, horizon_s=60.0, *, target="speed", test_seeds=(5,),
        val_seeds=(4,), lags=3, **knobs):
    """Score the scikit-learn trees exactly like :func:`ml.gbdt.run`."""
    runs, manifest = load_runs(data_dir)
    rest, test = split_runs(runs, test_seeds)
    train, val = split_runs(rest, val_seeds)
    h = max(1, round(horizon_s / manifest["bin_s"]))
    model = SKTreesFlowModel(h, manifest["day_length"], lags=lags,
                             target=target, **knobs).fit(train, val)
    rows = []
    for day in test:
        pred = model.predict_day(day)
        row = {"cars": int(day["cars"]), "car_seed": int(day["car_seed"]),
               "method": "sklearn trees", "target": target,
               "horizon_s": h * manifest["bin_s"], "n_trees": model.n_trees_}
        if target == "speed":
            row.update(evaluate_forecast(pred, day["speed"], day["counts"],
                                         day["edge_speed_limit"], h))
        else:
            row.update(evaluate_occupancy(pred, day["counts"], h))
        rows.append(row)
    return rows, model


def main(argv=None):
    """CLI: the scikit-learn backend on one dataset/target/horizon."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/varied")
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--target", choices=("speed", "counts"), default="speed")
    args = p.parse_args(argv)
    rows, model = run(args.data, horizon_s=args.horizon, target=args.target)
    key, unit = (("mae_kmh", "km/h") if args.target == "speed"
                 else ("mae_cars", "cars/street"))
    print(f"sklearn trees ({model.n_trees_} trees), target={args.target}, "
          f"horizon {args.horizon:.0f} s: "
          f"mean MAE {np.mean([r[key] for r in rows]):.4f} {unit}")


if __name__ == "__main__":
    main()
