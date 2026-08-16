"""Anchor tests for the optimization evaluation harness (``ml/opt/env.py``).

The harness is proven correct *before* any optimization runs, via two
regression anchors on a small-but-real default city:

1. **Identity**: applying the vector read from a fresh city reproduces the
   plain run bit-for-bit — the vector round-trip (read → apply → simulate)
   is lossless.
2. **Green wave**: the vector encoding of ``apply_green_wave``'s plans
   reproduces the ``--green-wave`` run bit-for-bit — a known non-trivial
   timing survives the encoding (this is what required offset wrapping).
"""

from experiments.common import build_default
from ml.opt.env import (DEFAULT_W, Evaluation, comparable, day_score,
                        evaluate, green_wave_vector, measure_baselines,
                        run_day, timing_space, wrap_offsets)
from traffic_sim.tuning import ParameterSpace

# A cheap but real day: the full default city, fewer cars, a fifth of a day.
SMALL = dict(cars=120, steps=2500)


def test_identity_vector_reproduces_baseline():
    plain = run_day(car_seed=1, **SMALL)
    space = timing_space(**SMALL)
    replay = run_day(vector=space.vector(), car_seed=1, **SMALL)
    assert comparable(replay) == comparable(plain)
    assert plain.get("trips_completed", 0) > 0  # the anchor isn't vacuous


def test_green_wave_vector_reproduces_green_wave():
    direct = run_day(car_seed=1, green_wave=True, **SMALL)
    encoded = run_day(vector=green_wave_vector(**SMALL), car_seed=1, **SMALL)
    assert comparable(encoded) == comparable(direct)
    # ... and the wave is a genuinely different timing, not a no-op.
    assert comparable(direct) != comparable(run_day(car_seed=1, **SMALL))


def test_wrap_offsets_is_modulo_cycle():
    space = timing_space(**SMALL)
    vec = space.vector()
    per = space._per_node
    cycle0 = (sum(vec[:space.num_phases])
              + space.num_phases * space.controller.plan_for(space.nodes[0]).yellow)
    vec[space.num_phases] = cycle0 * 2.5  # first node's offset, 2.5 cycles out
    wrapped = wrap_offsets(space, vec)
    assert abs(wrapped[space.num_phases] - cycle0 * 0.5) < 1e-9
    assert wrapped[per:] == vec[per:]  # other nodes untouched


def test_day_score_and_crash_cap():
    base = {"trips_completed": 100, "mean_delay_s": 50.0, "crashes": 1}
    assert abs(day_score(base, base) - (1 - DEFAULT_W)) < 1e-12
    better = {"trips_completed": 105, "mean_delay_s": 45.0, "crashes": 2}
    assert day_score(better, base) > day_score(base, base)
    crashy = {"trips_completed": 105, "mean_delay_s": 45.0, "crashes": 6}
    assert day_score(crashy, base) is None


def test_evaluate_identity_scores_one_minus_w():
    seeds = [1]
    base = measure_baselines(seeds, workers=1, **SMALL)
    ev = evaluate(timing_space(**SMALL).vector(), seeds, base,
                  workers=1, **SMALL)
    assert isinstance(ev, Evaluation) and not ev.rejected
    assert abs(ev.J - (1 - DEFAULT_W)) < 1e-12
