"""Tests for the scikit-learn-backed flow models (ml.sklearn_models)."""

import numpy as np

from ml.dataset import generate
from ml.linear import LinearFlowModel
from ml.sklearn_models import SKRidgeFlowModel, SKTreesFlowModel, run

TINY = dict(width=8, height=8, day_length=30.0,
            grade=False, ring=False, roundabouts=0,
            schedule=False, parking=False, demand=False)


def test_sk_ridge_agrees_with_our_closed_form(tmp_path):
    # Same features, same regularization idea, different solver: the two
    # ridge implementations must produce nearly identical predictions.
    generate(tmp_path, loads=(15,), seeds=(1, 2), bin_s=5.0, workers=1, **TINY)
    from ml.baselines import load_runs, split_runs
    runs, manifest = load_runs(tmp_path)
    train, test = split_runs(runs, test_seeds=(2,))
    ours = LinearFlowModel(1, manifest["day_length"], lags=2).fit(train)
    theirs = SKRidgeFlowModel(1, manifest["day_length"], lags=2).fit(train)
    a, b = ours.predict_day(test[0]), theirs.predict_day(test[0])
    assert np.abs(a - b).mean() < 0.1        # [m/s] — essentially the same


def test_sk_trees_end_to_end_both_targets(tmp_path):
    generate(tmp_path, loads=(10, 20), seeds=(1, 2, 3), bin_s=5.0, workers=1,
             **TINY)
    for target, key in (("speed", "mae_kmh"), ("counts", "mae_cars")):
        rows, model = run(tmp_path, horizon_s=5.0, target=target,
                          test_seeds=(3,), val_seeds=(2,), lags=2,
                          max_trees=50, step=10, patience_steps=2)
        assert rows and all(np.isfinite(r[key]) for r in rows)
        assert 10 <= model.n_trees_ <= 50


def test_sk_trees_learn_the_AND_rule():
    # The same interaction test the from-scratch trees pass.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(4000, 2))
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(float)
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=60, learning_rate=0.3,
                                      max_depth=3, random_state=0,
                                      early_stopping=False).fit(X, y)
    assert np.abs(m.predict(X) - y).mean() < 0.05
