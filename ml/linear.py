"""The first learned flow model: ridge linear regression, numpy only.

Deliberately the simplest genuine model (see the ML manual's "complexity must
pay rent" principle): the prediction is a weighted sum of features, one weight
per feature, fitted in closed form. If this cannot beat climatology, the
features are wrong and a fancier model would only hide that.

Features per (bin, edge) sample — everything known ``horizon`` before the
target bin, nothing after (no leakage by construction):

* **Lagged state** (``lags`` most recent bins up to ``b - horizon_bins``): the
  street's speed (empty → its speed limit), its occupancy, and an
  *empty-street flag* — so the model can learn that "no measurement" is not
  "free flow" (the failure that cripples persistence).
* **Climatology as a feature**: the per-(bin, edge) training-day mean — the
  strongest baseline, handed to the model so it only has to learn *when to
  trust today's state instead of the rhythm* (fitted on training days only).
* **Today's city-wide busyness**: the ratio of the day's cumulative city-wide
  vehicle count so far to the training-day average at the same clock time —
  a single global number a street's own lags cannot provide (a quiet Sunday
  is visible city-wide long before any one street proves it), plus its
  *product* with the climatology feature, so the model can express
  "climatology scaled by how busy today is" exactly.
* **Time of day** as sin/cos of the day angle (circular, so 23:59 and 00:01
  are neighbours).
* **Static street facts**: speed limit, length, lanes, elevated flag, and a
  one-hot land-use zone.

Features are standardized before the ridge solve; the solve is the normal
equation ``(X'X + ridge*I) w = X'y``. Evaluation reuses the baselines'
:func:`~ml.baselines.evaluate_forecast`, so the resulting row drops straight
into the same table.

Run (after generating a dataset)::

    MPLBACKEND=Agg bin/python -m ml.linear --data ml/data/default
"""

import argparse

import numpy as np

from ml.baselines import (climatology, evaluate_forecast, load_runs,
                          split_runs)

N_ZONES = 5   # 4 land-use codes + "unzoned" (code -1)


class LinearFlowModel:
    """Ridge regression from lagged street state to speed ``horizon`` ahead."""

    def __init__(self, horizon_bins: int, day_length: float, *,
                 lags: int = 3, ridge: float = 1.0,
                 target: str = "speed") -> None:
        """``horizon_bins`` is the forecast gap in bins; ``lags`` how many
        consecutive recent bins feed the model; ``ridge`` the regularization
        strength (on standardized features). ``day_length`` [s] sets the
        period of the time-of-day encoding. ``target`` is the channel to
        predict — ``"speed"`` [m/s] or ``"counts"`` (occupancy, cars on the
        street); the input features are identical, only the fitted output
        (and its climatology feature and physical clamp) change.
        """
        if target not in ("speed", "counts"):
            raise ValueError("target must be 'speed' or 'counts'")
        self.h = horizon_bins
        self.lags = lags
        self.ridge = ridge
        self.day_length = day_length
        self.target = target
        self.clim = None      # [n_bins, n_edges] training-day target means
        self.w = None         # fitted weights
        self._mu = None       # feature standardization
        self._sd = None

    # ------------------------------------------------------------- features

    def _static(self, run) -> np.ndarray:
        """Per-edge static features ``[n_edges, 8]`` (limit, length, lanes,
        elevated, zone one-hot)."""
        zone = run["edge_zone"].astype(int)
        onehot = np.zeros((zone.size, N_ZONES))
        onehot[np.arange(zone.size), np.where(zone < 0, N_ZONES - 1, zone)] = 1.0
        return np.column_stack([
            run["edge_speed_limit"], run["edge_length"] / 100.0,
            run["edge_lanes"].astype(float), (run["edge_level"] > 0).astype(float),
            onehot[:, :N_ZONES - 1],  # drop one column (redundant with bias)
        ])

    def _features(self, run, bins) -> np.ndarray:
        """Feature matrix for every (bin in ``bins``) x (every edge).

        Rows are ordered bin-major (all edges of bins[0], then bins[1], ...),
        matching ``run["speed"][bins].ravel()`` for the targets.
        """
        speed, counts = run["speed"], run["counts"]
        limit = run["edge_speed_limit"].astype(float)
        n_edges = speed.shape[1]
        static = self._static(run)
        # Today's cumulative city-wide count vs. the training-day average at
        # the same clock time — the observable "how busy is today" signal.
        city_cum = np.cumsum(counts.sum(axis=1))
        cols = []
        for b in bins:
            row_feats = []
            for lag in range(self.lags):
                s = speed[b - self.h - lag]
                empty = np.isnan(s)
                row_feats += [np.where(empty, limit, s),
                              counts[b - self.h - lag],
                              empty.astype(float)]
            row_feats.append(self.clim[b])
            busy = city_cum[b - self.h] / max(self._city_cum[b - self.h], 1e-9)
            row_feats.append(np.full(n_edges, busy))
            row_feats.append(self.clim[b] * busy)
            t = run["bin_t"][b]
            angle = 2.0 * np.pi * t / self.day_length
            row_feats.append(np.full(n_edges, np.sin(angle)))
            row_feats.append(np.full(n_edges, np.cos(angle)))
            cols.append(np.column_stack(row_feats + [static]))
        return np.concatenate(cols, axis=0)

    def _valid_bins(self, run) -> np.ndarray:
        """Bins with a full lag window available: ``b - h - lags + 1 >= 0``."""
        return np.arange(self.h + self.lags - 1, run["speed"].shape[0])

    def feature_names(self):
        """Human-readable name of each feature column, in matrix order.

        Must mirror :meth:`_features` exactly — used by the error-analysis
        step to label importance rankings.
        """
        from traffic_sim.zones import LandUse
        names = []
        for lag in range(self.lags):
            when = "latest" if lag == 0 else f"{lag} bin{'s' * (lag > 1)} earlier"
            names += [f"speed ({when})", f"count ({when})",
                      f"was-empty ({when})"]
        names += ["climatology", "city busyness", "climatology x busyness",
                  "time (sin)", "time (cos)",
                  "speed limit", "street length", "lanes", "elevated"]
        names += [f"zone: {u.name.lower()}" for u in list(LandUse)[:N_ZONES - 1]]
        return names

    # ------------------------------------------------------------ fit/predict

    def _fallback(self, run) -> np.ndarray:
        """Per-edge prediction for a cell with no information: free-flow speed
        for the speed target, an empty street for the counts target."""
        limit = run["edge_speed_limit"].astype(float)
        return limit if self.target == "speed" else np.zeros_like(limit)

    def _prepare(self, train_runs) -> None:
        """Precompute the training-day aggregates the features depend on."""
        self.clim = climatology([r[self.target] for r in train_runs],
                                self._fallback(train_runs[0]))
        self._city_cum = np.mean([np.cumsum(r["counts"].sum(axis=1))
                                  for r in train_runs], axis=0)

    def _matrix(self, runs):
        """``(X, y)`` over every defined-target cell of ``runs``."""
        xs, ys = [], []
        for run in runs:
            bins = self._valid_bins(run)
            X = self._features(run, bins)
            y = run[self.target][bins].ravel()
            keep = np.isfinite(y)
            xs.append(X[keep])
            ys.append(y[keep])
        return np.concatenate(xs), np.concatenate(ys)

    def fit(self, train_runs) -> "LinearFlowModel":
        """Fit weights on every defined-target cell of the training days."""
        self._prepare(train_runs)
        X, y = self._matrix(train_runs)

        self._mu = X.mean(axis=0)
        self._sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = np.column_stack([np.ones(len(X)), (X - self._mu) / self._sd])
        A = Xs.T @ Xs + self.ridge * np.eye(Xs.shape[1])
        A[0, 0] -= self.ridge          # don't penalize the bias term
        self.w = np.linalg.solve(A, Xs.T @ y)
        return self

    def _apply(self, X) -> np.ndarray:
        """Model output for a raw (unstandardized) feature matrix."""
        Xs = np.column_stack([np.ones(len(X)), (X - self._mu) / self._sd])
        return Xs @ self.w

    def predict_day(self, run) -> np.ndarray:
        """Predicted target ``[n_bins, n_edges]`` for one day.

        Bins without a full lag window fall back to the no-information
        prediction (free flow / empty street); they are outside the scored
        region anyway.
        """
        fallback = self._fallback(run)
        pred = np.tile(fallback, (run["speed"].shape[0], 1))
        bins = self._valid_bins(run)
        yhat = self._apply(self._features(run, bins))
        # Physical clamp: never negative; speeds also never beyond the limit.
        upper = (np.tile(run["edge_speed_limit"].astype(float), len(bins))
                 if self.target == "speed" else np.inf)
        yhat = np.clip(yhat, 0.0, upper)
        pred[bins] = yhat.reshape(len(bins), -1)
        return pred


def run(data_dir, horizon_s=60.0, test_seeds=(3,), lags=3, ridge=1.0):
    """Fit on the training days, score on the held-out days.

    Returns rows in the same schema as :func:`ml.baselines.run`, with
    ``method="linear"`` — plus the fitted model as the second return value
    (handy for inspection/plots).
    """
    runs, manifest = load_runs(data_dir)
    train, test = split_runs(runs, test_seeds)
    h = max(1, round(horizon_s / manifest["bin_s"]))
    model = LinearFlowModel(h, manifest["day_length"], lags=lags,
                            ridge=ridge).fit(train)
    rows = []
    for day in test:
        pred = model.predict_day(day)
        row = {"cars": int(day["cars"]), "car_seed": int(day["car_seed"]),
               "method": "linear", "horizon_s": h * manifest["bin_s"]}
        row.update(evaluate_forecast(pred, day["speed"], day["counts"],
                                     day["edge_speed_limit"], h))
        rows.append(row)
    return rows, model


def main(argv=None):
    """CLI: print the linear model's row next to the baselines."""
    from ml import baselines

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/default")
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--test-seeds", type=int, nargs="+", default=[3])
    p.add_argument("--lags", type=int, default=3)
    p.add_argument("--ridge", type=float, default=1.0)
    args = p.parse_args(argv)

    base_rows = baselines.run(args.data, horizon_s=args.horizon,
                              test_seeds=args.test_seeds)
    lin_rows, _ = run(args.data, horizon_s=args.horizon,
                      test_seeds=args.test_seeds, lags=args.lags,
                      ridge=args.ridge)
    rows = base_rows + lin_rows

    methods = ["persistence", "climatology", "climatology (load-matched)",
               "linear"]
    print(f"MAE [km/h] on held-out days (horizon {args.horizon:.0f} s):")
    print(f"{'method':<28}{'overall':>9}{'congested':>11}{'free-flow':>11}")
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        print(f"{m:<28}"
              f"{np.mean([r['mae_kmh'] for r in sel]):>9.2f}"
              f"{np.mean([r['mae_congested_kmh'] for r in sel]):>11.2f}"
              f"{np.mean([r['mae_freeflow_kmh'] for r in sel]):>11.2f}")


if __name__ == "__main__":
    main()
