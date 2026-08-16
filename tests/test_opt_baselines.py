"""Smoke test for the baseline measurement script (``ml/opt/baselines.py``)."""

from ml.opt.baselines import print_table, run, split_of


def test_run_smoke(tmp_path):
    rows = run(seeds=[1], workers=1, cars=120, steps=2500)
    assert len(rows) == 1
    r = rows[0]
    assert r["split"] == "train" and r["trips_base"] > 0
    assert r["J_gw"] is not None  # small day must not trip the crash cap
    print_table(rows)  # formatting must not raise (incl. the means block)


def test_split_labels():
    assert split_of(1) == "train"
    assert split_of(6) == "val"
    assert split_of(12) == "test"
    assert split_of(99) == "?"
