"""Tests for the flow-forecast baselines (ml.baselines)."""

import math

import numpy as np
import pytest

from ml.baselines import (climatology, evaluate_forecast, persistence, run,
                          split_runs)
from ml.dataset import generate
from traffic_sim.units import ms_to_kmh

NAN = float("nan")


def test_persistence_shifts_and_falls_back():
    speed = np.array([[10.0, NAN],
                      [11.0, NAN],
                      [12.0, 5.0],
                      [13.0, 6.0]])
    fallback = np.array([20.0, 8.0])
    pred = persistence(speed, horizon_bins=2, fallback=fallback)
    # Rows 2-3 echo rows 0-1; the empty column falls back to free flow.
    assert pred[2].tolist() == [10.0, 8.0]
    assert pred[3].tolist() == [11.0, 8.0]
    # Warm-up rows (nothing to persist from) are the fallback.
    assert pred[0].tolist() == [20.0, 8.0]


def test_climatology_averages_ignoring_empty_days():
    day1 = np.array([[10.0, NAN]])
    day2 = np.array([[14.0, NAN]])
    fallback = np.array([99.0, 8.0])
    clim = climatology([day1, day2], fallback)
    assert clim[0, 0] == 12.0     # mean over the days that measured something
    assert clim[0, 1] == 8.0      # empty on every training day -> free flow


def test_evaluate_scores_only_defined_targets_and_slices_regimes():
    target = np.array([[10.0, 10.0],   # bin 0: excluded (warm-up)
                       [4.0, NAN],     # congested cell + undefined cell
                       [9.0, 10.0]])   # free-flow cells
    pred = np.full_like(target, 8.0)
    counts = np.ones_like(target)
    limit = np.array([10.0, 10.0])
    out = evaluate_forecast(pred, target, counts, limit, horizon_bins=1)
    assert out["cells"] == 3                       # NaN target dropped
    assert math.isclose(out["mae_kmh"], ms_to_kmh((4 + 1 + 2) / 3))
    assert math.isclose(out["mae_congested_kmh"], ms_to_kmh(4.0))
    assert math.isclose(out["mae_freeflow_kmh"], ms_to_kmh(1.5))
    assert math.isclose(out["congested_share"], 1 / 3)


def test_split_is_by_whole_days():
    runs = [{"car_seed": s, "cars": c} for s in (1, 2, 3) for c in (10, 20)]
    train, test = split_runs(runs, test_seeds=(3,))
    assert len(train) == 4 and len(test) == 2
    assert all(r["car_seed"] != 3 for r in train)
    with pytest.raises(ValueError):
        split_runs(runs, test_seeds=(9,))


def test_end_to_end_on_tiny_dataset(tmp_path):
    tiny = dict(width=8, height=8, day_length=30.0,
                grade=False, ring=False, roundabouts=0,
                schedule=False, parking=False, demand=False)
    generate(tmp_path, loads=(10, 20), seeds=(1, 2), bin_s=5.0, workers=1,
             **tiny)
    rows = run(tmp_path, horizon_s=5.0, test_seeds=(2,))
    # 2 test days x 3 methods, each with a finite overall MAE.
    assert len(rows) == 6
    assert {r["method"] for r in rows} == {
        "persistence", "climatology", "climatology (load-matched)"}
    for r in rows:
        assert np.isfinite(r["mae_kmh"]) and r["cells"] > 0
