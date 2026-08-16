"""Smoke + consistency tests for the stage-1 structured search."""

from ml.opt.env import comparable, run_day
from ml.opt.stage1 import (CLASSICAL, run, stage1_context, structured_vector,
                           top_rows)

SMALL = dict(cars=120, steps=2500)


def test_classical_knobs_reproduce_green_wave():
    """The (1, 0.5, 1, 1) combo IS the green wave, bit-for-bit."""
    space, base_offsets, node_fast = stage1_context(**SMALL)
    vec = structured_vector(space, base_offsets, node_fast, **CLASSICAL)
    encoded = run_day(vector=vec, car_seed=1, **SMALL)
    direct = run_day(car_seed=1, green_wave=True, **SMALL)
    assert comparable(encoded) == comparable(direct)


def test_wave_zero_means_no_offsets():
    space, base_offsets, node_fast = stage1_context(**SMALL)
    vec = structured_vector(space, base_offsets, node_fast, wave=0.0)
    per = space._per_node
    offsets = [vec[k * per + space.num_phases] for k in range(len(space.nodes))]
    assert all(o == 0.0 for o in offsets)


def test_run_smoke():
    tiny_grid = {"cycle": (1.0,), "through": (0.5,), "bias": (1.0,),
                 "wave": (0.0, 1.0)}
    rows = run(grid=tiny_grid, seeds=[1], workers=1, log=lambda *a: None,
               **SMALL)
    assert len(rows) == 2
    assert all(r["J"] is not None for r in rows)
    assert top_rows(rows, 1)[0]["J"] == max(r["J"] for r in rows)
