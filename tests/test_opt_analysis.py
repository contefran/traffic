"""Verification of the stage-2 analysis module — all sim-free.

Row math, drift splitting, and both figure renders are exercised on
synthetic data shaped exactly like the real artifacts, so the tests run in
seconds and the module's plumbing is proven before it touches the (slow)
real evaluation days.
"""

import json
import os

import pytest

from ml.opt.analysis import (_row, drift_stats, mean_gain, render_curve,
                             render_final)
from ml.opt.env import DEFAULT_W, day_score

BASE = {"trips_completed": 1000, "mean_delay_s": 20.0, "crashes": 1}
GOOD = {"trips_completed": 1100, "mean_delay_s": 18.0, "crashes": 2,
        "aborted": False}
BAD = {"trips_completed": 900, "mean_delay_s": 30.0, "crashes": 9,
       "aborted": False}


def test_row_math_and_mean_gain():
    good = _row(8, BASE, GOOD, day_score(GOOD, BASE), DEFAULT_W)
    # trips ratio 1.1, delay ratio 0.9 -> J = 1.1 - 0.5*0.9 = 0.65 -> +0.15
    assert good["J_gain"] == pytest.approx(0.15)
    assert good["crashes"] == 2 and good["base_trips"] == 1000
    bad = _row(9, BASE, BAD, day_score(BAD, BASE), DEFAULT_W)
    assert bad["J_gain"] is None  # crash cap
    assert mean_gain([good, good]) == pytest.approx(0.15)
    assert mean_gain([good, bad]) is None


def test_drift_stats_splits_by_family():
    labels = ["n1.green0", "n1.green1", "n1.offset", "n2.green0"]
    warm = [10.0, 20.0, 5.0, 30.0]
    mean = [11.0, 18.0, 45.0, 30.0]
    d = drift_stats(mean, warm, labels)
    assert d["greens"] == [1.0, 2.0, 0.0]
    assert d["offsets"] == [40.0]
    assert d["greens_mean_s"] == pytest.approx(1.0)
    assert d["offsets_mean_s"] == pytest.approx(40.0)


def _fake_run():
    hist = [{"generation": g, "mean_score": 0.63 + 0.001 * g,
             "best_in_gen": 0.65, "best_so_far": 0.65, "rejected": 1}
            for g in range(6)]
    val = [{"generation": -1, "J": 0.663}, {"generation": 2, "J": None},
           {"generation": 4, "J": 0.645}]
    return {"history": hist, "val_history": val,
            "result": {"final_val_J_gain": 0.14, "val_history": val}}


def test_render_curve_smoke(tmp_path):
    drift = {"greens": [0.5, 1.0], "offsets": [2.0],
             "greens_mean_s": 0.75, "offsets_mean_s": 2.0}
    path = str(tmp_path / "curve.png")
    render_curve(_fake_run(), drift, path,
                 anchors={"train_J_gain": 0.167, "val_J_gain": 0.163})
    assert os.path.getsize(path) > 0


def test_render_final_smoke(tmp_path):
    good = _row(8, BASE, GOOD, day_score(GOOD, BASE), DEFAULT_W)
    board = {"green wave": [good, dict(good, seed=9)],
             "tuned plan": [dict(good, seed=8), dict(good, seed=9)]}
    load_rows = [dict(good, cars=c, seed=s)
                 for c in (500, 1000, 1500) for s in (8, 9)]
    path = str(tmp_path / "final.png")
    render_final(board, load_rows, path)
    assert os.path.getsize(path) > 0
