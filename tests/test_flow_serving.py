"""Tests for the serving path (ml.artifact + ml.serve).

The property that matters end-to-end: a forecast served over HTTP is
bit-identical to what the in-memory model computes — save/load and the web
layer add zero drift (the training–serving-skew guarantee).
"""

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.artifact import attach_geometry, cell_key, fit_bundle, load, save
from ml.baselines import load_runs, split_runs
from ml.dataset import generate
from ml.serve import create_app

TINY = dict(width=8, height=8, day_length=30.0,
            grade=False, ring=False, roundabouts=0,
            schedule=False, parking=False, demand=False)
CELLS = (("speed", 5.0), ("counts", 5.0))


@pytest.fixture(scope="module")
def bundle_env(tmp_path_factory):
    """One tiny dataset + fitted/saved/reloaded bundle for all tests."""
    root = tmp_path_factory.mktemp("serving")
    data = root / "data"
    generate(data, loads=(10, 20), seeds=(1, 2, 3), bin_s=5.0, workers=1,
             **TINY)
    bundle = fit_bundle(data, cells=CELLS, test_seeds=(3,), val_seeds=(2,),
                        max_trees=25, step=25)
    path = root / "bundle.joblib"
    save(bundle, path)
    runs, manifest = load_runs(data)
    _, test = split_runs(runs, (3,))
    return bundle, path, test[0], manifest


def _history(day, n_bins):
    """Day-so-far dict: the first ``n_bins`` bins plus the static arrays."""
    return {k: (v[:n_bins] if k in ("speed", "counts") else v)
            for k, v in day.items()}


def test_bundle_reloads_identically(bundle_env):
    bundle, path, day, _ = bundle_env
    reloaded = load(path)
    assert set(reloaded["models"]) == {cell_key(t, h) for t, h in CELLS}
    for key, model in bundle["models"].items():
        a = model.predict_day(day)
        b = reloaded["models"][key].predict_day(day)
        assert np.array_equal(a, b, equal_nan=True)   # bit-identical


def test_predict_next_matches_predict_day(bundle_env):
    bundle, _, day, _ = bundle_env
    for key, model in bundle["models"].items():
        full = model.predict_day(day)
        b_now = 4                                     # h=1, lags=3 -> valid
        got = model.predict_next(_history(day, b_now + 1))
        assert np.allclose(got, full[b_now + model.h], equal_nan=True)


def test_api_serves_the_model_exactly(bundle_env):
    bundle, path, day, manifest = bundle_env
    client = TestClient(create_app(str(path)))

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert {c["target"] for c in health["cells"]} == {"speed", "counts"}
    assert all(np.isfinite(c["test_mae"]) for c in health["cells"])

    b_now = 4
    speed = day["speed"][:b_now + 1]
    counts = day["counts"][:b_now + 1]
    body = {
        "target": "speed", "horizon_s": 5.0,
        "speed": [[None if np.isnan(x) else float(x) for x in row]
                  for row in speed],
        "counts": counts.tolist(),
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    out = resp.json()
    model = bundle["models"][cell_key("speed", 5.0)]
    want = model.predict_next(_history(day, b_now + 1))
    assert np.allclose(out["predictions"], want, atol=1e-4)  # 4-dp rounding
    assert out["t_forecast"] == out["t_observed"] + 5.0


def test_api_rejects_bad_requests(bundle_env):
    _, path, day, _ = bundle_env
    client = TestClient(create_app(str(path)))
    ok_speed = [[0.0] * day["speed"].shape[1]] * 5
    ok_counts = [[0.0] * day["speed"].shape[1]] * 5

    r = client.post("/predict", json={"target": "speed", "horizon_s": 999.0,
                                      "speed": ok_speed, "counts": ok_counts})
    assert r.status_code == 404                       # unknown cell

    r = client.post("/predict", json={"target": "speed", "horizon_s": 5.0,
                                      "speed": ok_speed[:2],
                                      "counts": ok_counts[:2]})
    assert r.status_code == 400                       # too few bins for lags

    r = client.post("/predict", json={"target": "speed", "horizon_s": 5.0,
                                      "speed": [[0.0, 1.0]] * 5,
                                      "counts": [[0.0, 1.0]] * 5})
    assert r.status_code == 400                       # wrong edge count


def test_network_endpoint_serves_drawable_geometry(bundle_env):
    bundle, path, day, manifest = bundle_env
    client = TestClient(create_app(str(path)))
    r = client.get("/network")
    assert r.status_code == 200
    net = r.json()
    assert net["n_edges"] == day["speed"].shape[1] == len(net["edges"])
    assert set(net["zone_codes"]) == set(manifest["zone_codes"])
    b = net["bounds"]
    assert b["x_min"] < b["x_max"] and b["y_min"] < b["y_max"]
    # Streets are straight segments between their endpoint nodes, so the
    # drawn length must reproduce the stored street length.
    for e in net["edges"]:
        drawn = math.hypot(e["x2"] - e["x1"], e["y2"] - e["y1"])
        assert drawn == pytest.approx(e["length"], rel=1e-3, abs=0.02)


def test_attach_geometry_rejects_wrong_city(bundle_env):
    from traffic_sim.network import build_grid_network

    bundle, _, _, _ = bundle_env
    with pytest.raises(ValueError, match="network mismatch"):
        attach_geometry(bundle, net=build_grid_network(3, 3, 100.0))


def test_index_serves_frontend_page(bundle_env):
    _, path, _, _ = bundle_env
    client = TestClient(create_app(str(path)))
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    # The page is a client of the JSON endpoints it draws from.
    assert "canvas" in r.text and "network" in r.text
    assert "health" in r.text and "demo/day" in r.text


def test_demo_day_replay_payload(bundle_env):
    bundle, path, day, manifest = bundle_env
    npz = Path(bundle["meta"]["data_dir"]) / manifest["runs"][0]["file"]
    client = TestClient(create_app(str(path), demo_day=str(npz)))
    d = client.get("/demo/day").json()
    assert d["n_bins"] == len(d["speed"]) == len(d["counts"]) > 0
    assert len(d["speed"][0]) == bundle["meta"]["n_edges"]
    # NaN (nobody drove) must survive the JSON trip as null, distinct
    # from 0 (standing traffic).
    assert any(x is None for row in d["speed"] for x in row)
    assert all(x is not None for row in d["counts"] for x in row)
    # The payload must round-trip straight back into /predict.
    b_now = 4
    r = client.post("/predict", json={
        "target": "speed", "horizon_s": 5.0,
        "speed": d["speed"][:b_now + 1], "counts": d["counts"][:b_now + 1]})
    assert r.status_code == 200


def test_demo_day_404_when_missing(bundle_env, tmp_path):
    _, path, _, _ = bundle_env
    client = TestClient(create_app(str(path),
                                   demo_day=str(tmp_path / "nope.npz")))
    r = client.get("/demo/day")
    assert r.status_code == 404
    assert "FLOW_DEMO_DAY" in r.json()["detail"]


def test_network_404_without_geometry(bundle_env, tmp_path):
    bundle, _, _, _ = bundle_env
    bare = {**bundle,
            "statics": {k: v for k, v in bundle["statics"].items()
                        if not k.startswith("node_")}}
    p = tmp_path / "bare.joblib"
    save(bare, p)
    client = TestClient(create_app(str(p)))
    r = client.get("/network")
    assert r.status_code == 404
    assert "attach-geometry" in r.json()["detail"]
