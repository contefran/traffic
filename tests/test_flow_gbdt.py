"""Tests for the from-scratch boosted-trees flow model (ml.gbdt)."""

import numpy as np

from ml.dataset import generate
from ml.gbdt import GBDTFlowModel, _fit_tree, run


def test_single_tree_finds_an_obvious_split():
    # One feature, residuals +1 for bins above 10, -1 below: the best split
    # is at bin 10 and the leaves must predict the two means.
    Xb = np.repeat(np.arange(32, dtype=np.uint8), 20)[:, None]
    grad = np.where(Xb[:, 0] > 10, 1.0, -1.0)
    tree = _fit_tree(Xb, grad, max_depth=1, min_leaf=5, n_bins=32)
    pred = tree.predict(Xb)
    assert np.allclose(pred[Xb[:, 0] <= 10], -1.0)
    assert np.allclose(pred[Xb[:, 0] > 10], 1.0)


def test_boosting_learns_an_AND_rule_no_linear_model_can():
    # Target = 1 only when feature A AND feature B are both high — the
    # classic interaction a weighted sum cannot express but a two-question
    # flowchart answers trivially.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(4000, 2))
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(float)

    model = GBDTFlowModel(horizon_bins=1, day_length=100.0, n_trees=60,
                          learning_rate=0.3, max_depth=3, min_leaf=20,
                          subsample=1.0, seed=1)
    model._fit_bins(X)
    Xb = model._bin(X)
    pred = np.full(len(y), y.mean())
    for _ in range(model.n_trees):
        tree = _fit_tree(Xb, y - pred, model.max_depth, model.min_leaf,
                         model.n_bins)
        pred += model.learning_rate * tree.predict(Xb)
    assert np.abs(pred - y).mean() < 0.05

    # A pure weighted sum on the same features stays far from the truth.
    Xs = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.lstsq(Xs, y, rcond=None)[0]
    assert np.abs(Xs @ w - y).mean() > 0.2


def test_end_to_end_with_early_stopping(tmp_path):
    tiny = dict(width=8, height=8, day_length=30.0,
                grade=False, ring=False, roundabouts=0,
                schedule=False, parking=False, demand=False)
    generate(tmp_path, loads=(10, 20), seeds=(1, 2, 3), bin_s=5.0, workers=1,
             **tiny)
    for target, key in (("speed", "mae_kmh"), ("counts", "mae_cars")):
        rows, model = run(tmp_path, horizon_s=5.0, target=target,
                          test_seeds=(3,), val_seeds=(2,), lags=2,
                          n_trees=30, min_leaf=5, patience=5)
        assert rows and all(np.isfinite(r[key]) for r in rows)
        assert 1 <= len(model.trees) <= 30      # early stopping kept a prefix


def test_deterministic_given_seed(tmp_path):
    tiny = dict(width=8, height=8, day_length=30.0,
                grade=False, ring=False, roundabouts=0,
                schedule=False, parking=False, demand=False)
    generate(tmp_path, loads=(10,), seeds=(1, 2, 3), bin_s=5.0, workers=1,
             **tiny)
    a = run(tmp_path, horizon_s=5.0, test_seeds=(3,), val_seeds=(2,),
            lags=2, n_trees=10, min_leaf=5, seed=7)[0]
    b = run(tmp_path, horizon_s=5.0, test_seeds=(3,), val_seeds=(2,),
            lags=2, n_trees=10, min_leaf=5, seed=7)[0]
    assert [r["mae_kmh"] for r in a] == [r["mae_kmh"] for r in b]
