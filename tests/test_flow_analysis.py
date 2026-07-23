"""Smoke tests for the error-analysis step (ml.analysis)."""

import numpy as np

from ml.analysis import analyze, street_classes
from ml.dataset import generate

TINY = dict(width=8, height=8, day_length=30.0,
            grade=False, ring=False, roundabouts=0,
            schedule=False, parking=False, demand=False)


def test_analyze_produces_all_slices(tmp_path):
    generate(tmp_path, loads=(10, 20), seeds=(1, 2, 3), bin_s=5.0, workers=1,
             **TINY)
    res = analyze(tmp_path, target="counts", horizon_s=5.0,
                  test_seeds=(3,), val_seeds=(2,), lags=2,
                  max_trees=20, step=10, patience_steps=1, n_repeats=2)
    names = {n for n, _ in res["importances"]}
    assert len(res["importances"]) == len(res["model"].feature_names())
    assert "climatology" in names and "city busyness" in names
    assert res["hourly_mae"].shape == (24,)
    assert np.nanmean(res["hourly_mae"]) >= 0
    assert set(res["class_mae"]) <= set(street_classes({
        "edge_speed_limit": np.array([10.0]), "edge_level": np.array([0])}))
    assert len(res["worst_edges"]) == 20
    # Worst edges really are sorted by error, descending.
    vals = [v for _, v in res["worst_edges"]]
    assert vals == sorted(vals, reverse=True)
