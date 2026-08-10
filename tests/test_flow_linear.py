"""Tests for the linear flow model (ml.linear)."""

import numpy as np

from ml.dataset import generate
from ml.linear import LinearFlowModel, run


def _fake_run(const_speeds, n_bins=30, bin_s=5.0):
    """A synthetic day where every edge holds one constant speed all day.

    Topology is a simple one-way chain (edge e runs node e -> e+1), so the
    neighbour features are well-defined: each edge's feeder is the previous
    chain link, its receiver the next.
    """
    n_edges = len(const_speeds)
    speed = np.tile(np.asarray(const_speeds, float), (n_bins, 1))
    return {
        "speed": speed.astype(np.float32),
        "counts": np.ones((n_bins, n_edges), np.float32),
        "bin_t": (np.arange(n_bins) * bin_s).astype(np.float32),
        "edge_speed_limit": np.full(n_edges, 15.0, np.float32),
        "edge_length": np.full(n_edges, 100.0, np.float32),
        "edge_lanes": np.ones(n_edges, np.int16),
        "edge_level": np.zeros(n_edges, np.int16),
        "edge_zone": np.zeros(n_edges, np.int16),
        "edge_u": np.arange(n_edges, dtype=np.int32),
        "edge_v": np.arange(1, n_edges + 1, dtype=np.int32),
    }


def test_learns_persistence_when_it_is_the_truth():
    # Each day every street holds a different constant speed, so "recent
    # speed" predicts the target perfectly while climatology (the cross-day
    # mean) cannot. The fitted model must pick up the lag feature and get
    # near-zero error on an unseen day.
    train = [_fake_run([4.0, 6.0, 8.0, 10.0, 12.0]),
             _fake_run([12.0, 10.0, 8.0, 6.0, 4.0])]
    test = _fake_run([5.0, 11.0, 7.0, 13.0, 3.0])

    model = LinearFlowModel(horizon_bins=2, day_length=150.0,
                            lags=2, ridge=1e-6).fit(train)
    pred = model.predict_day(test)
    scored = model._valid_bins(test)
    err = np.abs(pred[scored] - test["speed"][scored])
    assert err.mean() < 0.05                      # essentially exact
    clim_err = np.abs(model.clim[scored] - test["speed"][scored])
    assert err.mean() < 0.1 * clim_err.mean()     # and far beyond climatology


def test_prediction_respects_physical_bounds():
    train = [_fake_run([4.0, 6.0, 8.0, 10.0, 12.0])]
    model = LinearFlowModel(horizon_bins=2, day_length=150.0,
                            lags=2, ridge=1.0).fit(train)
    pred = model.predict_day(_fake_run([5.0, 11.0, 7.0, 13.0, 3.0]))
    assert (pred >= 0.0).all()
    assert (pred <= 15.0 + 1e-6).all()


def test_counts_target_learns_and_stays_nonnegative():
    # Same constant-per-day construction, but the truth lives in the counts
    # channel: occupancy is constant per edge per day, so the lagged count
    # must predict it near-exactly on an unseen day.
    train, test = [], _fake_run([5.0] * 5)
    for k, base in enumerate((0.5, 2.0)):
        r = _fake_run([5.0] * 5)
        r["counts"] = np.full_like(r["counts"], base + k)
        train.append(r)
    test["counts"] = np.full_like(test["counts"], 1.7)

    model = LinearFlowModel(horizon_bins=2, day_length=150.0, lags=2,
                            ridge=1e-6, target="counts").fit(train)
    pred = model.predict_day(test)
    scored = model._valid_bins(test)
    assert np.abs(pred[scored] - 1.7).mean() < 0.05
    assert (pred >= 0.0).all()


def test_end_to_end_on_tiny_dataset(tmp_path):
    tiny = dict(width=8, height=8, day_length=30.0,
                grade=False, ring=False, roundabouts=0,
                schedule=False, parking=False, demand=False)
    generate(tmp_path, loads=(10, 20), seeds=(1, 2), bin_s=5.0, workers=1,
             **tiny)
    rows, model = run(tmp_path, horizon_s=5.0, test_seeds=(2,), lags=2)
    assert len(rows) == 2 and all(r["method"] == "linear" for r in rows)
    for r in rows:
        assert np.isfinite(r["mae_kmh"]) and r["cells"] > 0
    assert model.w is not None


def test_neighbour_features_follow_topology():
    # A 4-link chain (0->1->2->3->4): edge 1's only feeder is edge 0, its
    # only receiver edge 2; the chain ends fall back to free flow / empty.
    run_ = _fake_run([4.0, 6.0, 8.0, 10.0])
    run_["counts"] = np.tile(np.array([1.0, 2.0, 3.0, 4.0], np.float32),
                             (run_["counts"].shape[0], 1))
    model = LinearFlowModel(horizon_bins=1, day_length=150.0, lags=1)
    model.clim = np.zeros_like(run_["speed"])  # unused by the helper
    nb = model._neighbour_feats(run_, b=5)
    up_speed, up_cnt, up_max, dn_speed, dn_cnt, dn_max = nb.T
    assert np.allclose(up_speed, [15.0, 4.0, 6.0, 8.0])   # edge 0: no feeder
    assert np.allclose(up_cnt, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(up_max, up_cnt)                    # single feeders
    assert np.allclose(dn_speed, [6.0, 8.0, 10.0, 15.0])  # edge 3: no receiver
    assert np.allclose(dn_cnt, [2.0, 3.0, 4.0, 0.0])


def test_neighbour_features_exclude_reverse_edge():
    # A single two-way street (edges 0: A->B and 1: B->A) has no *genuine*
    # neighbours — each edge's only feeder/receiver is its own reverse, which
    # must not count as upstream/downstream traffic.
    run_ = _fake_run([4.0, 6.0])
    run_["edge_u"] = np.array([0, 1], np.int32)
    run_["edge_v"] = np.array([1, 0], np.int32)
    model = LinearFlowModel(horizon_bins=1, day_length=150.0, lags=1)
    nb = model._neighbour_feats(run_, b=5)
    assert np.allclose(nb[:, 0], 15.0)   # upstream speed -> free-flow fallback
    assert np.allclose(nb[:, 1], 0.0)    # upstream count -> empty
    assert np.allclose(nb[:, 3], 15.0)   # downstream speed
    assert np.allclose(nb[:, 4], 0.0)    # downstream count
