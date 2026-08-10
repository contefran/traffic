"""Gradient-boosted decision trees for the flow model, from scratch in numpy.

Plain-language recap (the ML manual has the long version): a *decision tree*
is a flowchart of yes/no questions about the features, ending in a predicted
number; *boosting* builds hundreds of small trees in sequence, each trained
only on the errors left over by the sum of the previous ones. The result can
express "IF rush hour AND arterial AND busy day THEN expect a queue" without
anyone hand-building that combination — the question the linear model cannot
answer without pre-made product features.

Implementation choices, all standard and all visible:

* **Histogram splits**: every feature is pre-quantized into at most
  ``n_bins`` quantile bins (an integer per cell), so finding a split is
  counting, not sorting. This is the same trick production libraries use.
* **Squared-error boosting**: each tree fits the current residuals; a leaf
  predicts the mean residual of its samples; predictions accumulate with a
  small ``learning_rate`` (many cautious corrections beat few bold ones).
* **Row subsampling** per tree — cheaper and a mild regularizer.
* **Early stopping on validation days**: training days grow the trees,
  *validation* days (whole held-out days, never test days) decide how many
  trees to keep — boosting overfits by adding trees, and the validation curve
  says when to stop.

The model shares :class:`~ml.linear.LinearFlowModel`'s entire feature
pipeline by subclassing it — identical inputs, identical evaluation, so any
difference in the table is the model and nothing else.

Run (after generating a dataset)::

    MPLBACKEND=Agg bin/python -m ml.gbdt --data ml/data/varied
"""

import argparse

import numpy as np

from ml.baselines import (evaluate_forecast, evaluate_occupancy, load_runs,
                          split_runs)
from ml.linear import LinearFlowModel


class _Tree:
    """One depth-limited regression tree on binned features (flat arrays)."""

    __slots__ = ("feature", "split", "left", "right", "value", "depth")

    def __init__(self, max_nodes: int, depth: int):
        self.feature = np.full(max_nodes, -1, np.int32)   # -1 = leaf
        self.split = np.zeros(max_nodes, np.int32)        # go left if bin <= split
        self.left = np.zeros(max_nodes, np.int32)
        self.right = np.zeros(max_nodes, np.int32)
        self.value = np.zeros(max_nodes, np.float64)
        self.depth = depth

    def predict(self, Xb: np.ndarray) -> np.ndarray:
        """Route every row down the flowchart to its leaf value."""
        node = np.zeros(len(Xb), np.int32)
        rows = np.arange(len(Xb))
        for _ in range(self.depth + 1):
            f = self.feature[node]
            active = f >= 0
            if not active.any():
                break
            n = node[active]
            goes_left = Xb[rows[active], f[active]] <= self.split[n]
            node[active] = np.where(goes_left, self.left[n], self.right[n])
        return self.value[node]


def _fit_tree(Xb, grad, max_depth, min_leaf, n_bins) -> _Tree:
    """Grow one tree on binned features ``Xb`` fitting the residuals ``grad``.

    A split's *gain* is how much it reduces squared error, computed for every
    candidate (feature, bin threshold) at once from per-bin histograms of
    (count, residual sum) — counting, not sorting.
    """
    n_feat = Xb.shape[1]
    tree = _Tree(2 ** (max_depth + 1), max_depth)
    tree.value[0] = grad.mean()
    n_nodes = 1
    stack = [(0, np.arange(len(grad)), 0)]      # (node id, row idx, depth)
    while stack:
        node, idx, depth = stack.pop()
        r = grad[idx]
        total_sum, total_n = r.sum(), len(idx)
        tree.value[node] = total_sum / total_n
        if depth >= max_depth or total_n < 2 * min_leaf:
            continue
        best_gain, best_f, best_b = 1e-12, -1, -1
        parent_score = total_sum * total_sum / total_n
        for f in range(n_feat):
            col = Xb[idx, f]
            cnt = np.bincount(col, minlength=n_bins)
            sm = np.bincount(col, weights=r, minlength=n_bins)
            cum_n = np.cumsum(cnt)[:-1]          # left side if split after bin b
            cum_s = np.cumsum(sm)[:-1]
            ok = (cum_n >= min_leaf) & (total_n - cum_n >= min_leaf)
            if not ok.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = (cum_s ** 2 / cum_n
                        + (total_sum - cum_s) ** 2 / (total_n - cum_n)
                        - parent_score)
            gain = np.where(ok, gain, -np.inf)
            b = int(np.argmax(gain))
            if gain[b] > best_gain:
                best_gain, best_f, best_b = gain[b], f, b
        if best_f < 0:
            continue
        goes_left = Xb[idx, best_f] <= best_b
        tree.feature[node] = best_f
        tree.split[node] = best_b
        tree.left[node] = n_nodes
        tree.right[node] = n_nodes + 1
        n_nodes += 2
        stack.append((tree.left[node], idx[goes_left], depth + 1))
        stack.append((tree.right[node], idx[~goes_left], depth + 1))
    return tree


class GBDTFlowModel(LinearFlowModel):
    """Boosted trees over the exact feature pipeline of the linear model."""

    def __init__(self, horizon_bins: int, day_length: float, *,
                 lags: int = 3, target: str = "speed", neighbours: bool = True,
                 n_trees: int = 300, learning_rate: float = 0.1,
                 max_depth: int = 5, min_leaf: int = 50,
                 subsample: float = 0.5, n_bins: int = 64,
                 patience: int = 25, seed: int = 0) -> None:
        """Boosting knobs on top of the shared feature setup.

        ``n_trees`` is a ceiling; with validation days the actual count is
        chosen by early stopping (``patience`` trees without improvement).
        """
        super().__init__(horizon_bins, day_length, lags=lags, target=target,
                         neighbours=neighbours)
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.subsample = subsample
        self.n_bins = n_bins
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.trees = []
        self.base = 0.0
        self._edges = None    # per-feature bin edges

    # ------------------------------------------------------------- binning

    def _bin(self, X) -> np.ndarray:
        """Quantize ``X`` columns into the fitted per-feature bins."""
        Xb = np.empty(X.shape, np.uint8)
        for j, edges in enumerate(self._edges):
            Xb[:, j] = np.searchsorted(edges, X[:, j]).astype(np.uint8)
        return Xb

    def _fit_bins(self, X) -> None:
        """Choose quantile bin edges per feature from (a sample of) X."""
        sample = X if len(X) <= 200_000 else \
            X[self.rng.choice(len(X), 200_000, replace=False)]
        qs = np.linspace(0, 1, self.n_bins)[1:-1]
        self._edges = [np.unique(np.quantile(sample[:, j], qs))
                       for j in range(X.shape[1])]

    # ---------------------------------------------------------- fit/predict

    def fit(self, train_runs, val_runs=None) -> "GBDTFlowModel":
        """Boost on the training days; ``val_runs`` (whole days) pick the
        stopping point. Without them, all ``n_trees`` are kept."""
        self._prepare(train_runs)
        X, y = self._matrix(train_runs)
        self._fit_bins(X)
        Xb = self._bin(X)
        del X

        self.base = float(y.mean())
        pred = np.full(len(y), self.base)

        if val_runs is not None:
            Xv, yv = self._matrix(val_runs)
            vb = self._bin(Xv)
            del Xv
            val_pred = np.full(len(yv), self.base)
        best_err, best_len, since = np.inf, 0, 0

        for _ in range(self.n_trees):
            rows = (np.arange(len(y)) if self.subsample >= 1.0 else
                    self.rng.choice(len(y), int(len(y) * self.subsample),
                                    replace=False))
            tree = _fit_tree(Xb[rows], (y - pred)[rows], self.max_depth,
                             self.min_leaf, self.n_bins)
            self.trees.append(tree)
            pred += self.learning_rate * tree.predict(Xb)
            if val_runs is None:
                continue
            val_pred += self.learning_rate * tree.predict(vb)
            err = np.abs(val_pred - yv).mean()
            if err < best_err - 1e-9:
                best_err, best_len, since = err, len(self.trees), 0
            else:
                since += 1
                if since >= self.patience:
                    break
        if val_runs is not None:
            self.trees = self.trees[:best_len]
        return self

    def _apply(self, X) -> np.ndarray:
        """Sum of the base value and every tree's correction."""
        Xb = self._bin(X)
        out = np.full(len(X), self.base)
        for tree in self.trees:
            out += self.learning_rate * tree.predict(Xb)
        return out


def run(data_dir, horizon_s=60.0, *, target="speed", test_seeds=(5,),
        val_seeds=(4,), lags=3, **knobs):
    """Fit (train days) + early-stop (validation days) + score (test days).

    Rows drop into the same table as :mod:`ml.baselines` / :mod:`ml.linear`;
    the fitted model comes back too.
    """
    runs, manifest = load_runs(data_dir)
    rest, test = split_runs(runs, test_seeds)
    train, val = split_runs(rest, val_seeds)
    h = max(1, round(horizon_s / manifest["bin_s"]))
    model = GBDTFlowModel(h, manifest["day_length"], lags=lags, target=target,
                          **knobs).fit(train, val)
    rows = []
    for day in test:
        pred = model.predict_day(day)
        row = {"cars": int(day["cars"]), "car_seed": int(day["car_seed"]),
               "method": "gbdt", "target": target,
               "horizon_s": h * manifest["bin_s"], "n_trees": len(model.trees)}
        if target == "speed":
            row.update(evaluate_forecast(pred, day["speed"], day["counts"],
                                         day["edge_speed_limit"], h))
        else:
            row.update(evaluate_occupancy(pred, day["counts"], h))
        rows.append(row)
    return rows, model


def main(argv=None):
    """CLI: boosted trees vs the linear model on one dataset and target."""
    from ml import linear as linear_mod

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="ml/data/varied")
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--target", choices=("speed", "counts"), default="speed")
    p.add_argument("--test-seeds", type=int, nargs="+", default=[5])
    p.add_argument("--val-seeds", type=int, nargs="+", default=[4])
    args = p.parse_args(argv)

    gb_rows, model = run(args.data, horizon_s=args.horizon, target=args.target,
                         test_seeds=tuple(args.test_seeds),
                         val_seeds=tuple(args.val_seeds))
    key, unit = (("mae_kmh", "km/h") if args.target == "speed"
                 else ("mae_cars", "cars/street"))
    print(f"gbdt ({len(model.trees)} trees), target={args.target}, "
          f"horizon {args.horizon:.0f} s:")
    for r in gb_rows:
        print(f"  cars={r['cars']:>5}: MAE {r[key]:.4f} {unit}")
    print(f"  mean: {np.mean([r[key] for r in gb_rows]):.4f} {unit}")


if __name__ == "__main__":
    main()
