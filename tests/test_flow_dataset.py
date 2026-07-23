"""Smoke tests for the flow-model dataset generator (ml.dataset)."""

import json

import numpy as np

from ml.dataset import generate, run_day

# Tiny default-city variant, mirroring tests/test_experiments.py: small plain
# grid, few cars, a short "day" — just enough to exercise the code path.
TINY_CITY = dict(width=8, height=8, day_length=30.0,
                 grade=False, ring=False, roundabouts=0,
                 schedule=False, parking=False, demand=False)


def test_run_day_shapes_and_values():
    data = run_day(cars=15, car_seed=1, bin_s=5.0, **TINY_CITY)
    counts, speed = data["counts"], data["speed"]
    n_edges = data["edge_length"].shape[0]
    assert counts.shape == speed.shape == (6, n_edges)  # 30 s day / 5 s bins
    assert data["bin_t"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    # Cars actually drove somewhere.
    assert counts.sum() > 0
    # Speed is NaN exactly where nothing drove, finite and sane elsewhere.
    empty = counts == 0
    assert np.isnan(speed[empty]).all()
    assert np.isfinite(speed[~empty]).all()
    assert (speed[~empty] >= 0).all()
    assert speed[~empty].max() <= data["edge_speed_limit"].max() + 1e-6
    # Static features cover every edge.
    for key in ("edge_speed_limit", "edge_lanes", "edge_level",
                "edge_zone", "edge_u", "edge_v"):
        assert data[key].shape == (n_edges,)


def test_generate_writes_runs_and_manifest(tmp_path):
    manifest = generate(tmp_path, loads=(10, 20), seeds=(1,), bin_s=5.0,
                        workers=1, **TINY_CITY)
    assert len(manifest["runs"]) == 2
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["bin_s"] == 5.0 and on_disk["day_length"] == 30.0

    # Columns line up across runs: same network, same edge order.
    a = np.load(tmp_path / manifest["runs"][0]["file"])
    b = np.load(tmp_path / manifest["runs"][1]["file"])
    assert a["counts"].shape == b["counts"].shape
    assert np.array_equal(a["edge_length"], b["edge_length"])
    assert np.array_equal(a["edge_zone"], b["edge_zone"])
