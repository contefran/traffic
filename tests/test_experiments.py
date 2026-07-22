"""Smoke tests for the experiment scripts (tiny sweeps, just exercise the code)."""

import os

from experiments import (q17_gap_sweep, q18_protected_vs_permissive,
                         q19_following_gap, q20_personality)

# A tiny default-city variant for the full-day experiments (q19/q20): small
# plain grid, few cars, a short "day", no elevated ring / roundabouts /
# schedules — just enough to exercise the code path.
TINY_CITY = dict(width=8, height=8, cars=25, day_length=60.0,
                 grade=False, ring=False, roundabouts=0,
                 schedule=False, parking=False, demand=False)


def test_q17_runs_and_renders(tmp_path):
    rows = q17_gap_sweep.run(gap_scales=(0.4, 1.0), n_cars=20, steps=80, seed=1)
    assert len(rows) == 2
    for r in rows:
        assert {"gap_scale", "throughput_per_s", "crashes"} <= r.keys()
    out = q17_gap_sweep.render(rows, path=str(tmp_path / "q17.png"))
    assert os.path.getsize(out) > 0


def test_q18_runs_and_renders(tmp_path):
    rows = q18_protected_vs_permissive.run(loads=(20, 40), steps=80,
                                           speeds_kmh=(50.0, 70.0), seeds=(1,),
                                           workers=1)
    assert len(rows) == 8  # 2 loads x 2 speeds x 2 controllers
    assert {r["controller"] for r in rows} == {"permissive", "protected"}
    assert {r["speed_kmh"] for r in rows} == {50.0, 70.0}
    out = q18_protected_vs_permissive.render(rows, path=str(tmp_path / "q18.png"))
    assert os.path.getsize(out) > 0


def test_q19_runs_and_renders(tmp_path):
    rows = q19_following_gap.run(gap_scales=(0.8, 1.5), seeds=(1, 2),
                                 workers=1, **TINY_CITY)
    assert len(rows) == 4  # 2 scales x 2 seeds
    for r in rows:
        assert {"gap_scale", "seed", "crashes", "fuel_proxy"} <= r.keys()
    out = q19_following_gap.render(rows, path=str(tmp_path / "q19.png"))
    assert os.path.getsize(out) > 0


def test_q20_groups_all_cars_and_renders(tmp_path):
    # A longer window than TINY_CITY's: trips must have time to complete.
    rows = q20_personality.run(seeds=(1,), workers=1,
                               **{**TINY_CITY, "day_length": 240.0})
    assert rows, "expected completed trips to aggregate"
    groupings = {r["grouping"] for r in rows}
    assert groupings == {"personality", "vtype"}
    # The personality terciles partition the fleet's trips.
    per = [r for r in rows if r["grouping"] == "personality"]
    vty = [r for r in rows if r["grouping"] == "vtype"]
    assert sum(r["trips"] for r in per) == sum(r["trips"] for r in vty)
    out = q20_personality.render(rows, path=str(tmp_path / "q20.png"))
    assert os.path.getsize(out) > 0
