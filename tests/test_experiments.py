"""Smoke tests for the experiment scripts (tiny sweeps, just exercise the code)."""

import os

from experiments import q17_gap_sweep, q18_protected_vs_permissive


def test_q17_runs_and_renders(tmp_path):
    rows = q17_gap_sweep.run(gap_scales=(0.4, 1.0), n_cars=20, steps=80, seed=1)
    assert len(rows) == 2
    for r in rows:
        assert {"gap_scale", "throughput_per_s", "crashes"} <= r.keys()
    out = q17_gap_sweep.render(rows, path=str(tmp_path / "q17.png"))
    assert os.path.getsize(out) > 0


def test_q18_runs_and_renders(tmp_path):
    rows = q18_protected_vs_permissive.run(loads=(20, 40), steps=80, seed=1)
    assert len(rows) == 4  # 2 loads x 2 controllers
    controllers = {r["controller"] for r in rows}
    assert controllers == {"permissive", "protected"}
    out = q18_protected_vs_permissive.render(rows, path=str(tmp_path / "q18.png"))
    assert os.path.getsize(out) > 0
