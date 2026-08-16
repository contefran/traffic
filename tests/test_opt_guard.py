"""Verification of the gridlock early-abort guard and wall-time recording.

The guard's job is to discard hopeless candidates *early and correctly*:
these tests check both halves — a good timing passes untouched (no false
positive, dynamics bit-identical to an unguarded run), and a deliberately
strangling timing is aborted at the checkpoint *and* proven to deserve it
(the full unguarded run confirms the collapse the early check predicted).

Timescale: the guard compares trips-so-far against the same-seed baseline at
the same *fraction* of the day, so the mechanism scales — but physical
travel time does not compress with ``day_length``, so a coherent compressed
day shrinks ``work_scale`` too (shorter commutes for a shorter day, the
`activities.py` coherence rule). The tests run a **compressed full day**
(``day_length=400`` → 4000 steps, both rushes included) with the checkpoint
at 0.5 (= noon), well after the morning rush has completed its trips.
"""

from ml.opt.env import (AbortGuard, comparable, day_score, evaluate,
                        measure_baselines, run_day, timing_space)

# A compressed but coherent day: shorter commutes to match the shorter day.
DAY = dict(cars=150, day_length=400.0, work_scale=150.0)
FRAC = 0.5  # checkpoint at noon of the compressed day
RECORD_ONLY = AbortGuard(frac=FRAC)  # empty baseline: records, never aborts


def _base():
    return measure_baselines([1], workers=1, abort=RECORD_ONLY, **DAY)


def _gridlock_vector(space):
    """A genuinely catastrophic timing: every phase at the 60 s maximum — a
    ~246 s cycle whose ~3-minute reds dwarf this day's ~80 s trips, so cars
    spend the rush parked at lights (probed: checkpoint ratio 0.22, total
    trips 110 vs baseline 325). NB a *milder* disaster (starving one
    orientation) probes at ratio 0.41 — bad but scoreable, and correctly
    NOT aborted: the guard is a catastrophe filter, not a bad-candidate
    filter."""
    lo, hi = space.green_bounds
    vec = []
    for _ in space.nodes:
        vec += [hi] * space.num_phases + [0.0]
    vec += [1.0] * len(space.group_names)
    return vec


def test_wall_time_recorded_dynamics_still_deterministic():
    a = run_day(car_seed=1, **DAY)
    b = run_day(car_seed=1, **DAY)
    assert a["wall_s"] > 0 and b["wall_s"] > 0
    assert comparable(a) == comparable(b)  # only the wall clock may differ


def test_baseline_records_checkpoint_trips():
    base = _base()
    assert base[1]["trips_at_frac"] > 0  # checkpoint sees completed trips
    assert not base[1]["aborted"]
    guard = AbortGuard.from_baselines(base, frac=FRAC)
    assert guard.baseline == {1: base[1]["trips_at_frac"]}
    assert guard.frac == FRAC


def test_guard_passes_a_good_candidate_untouched():
    base = _base()
    guard = AbortGuard.from_baselines(base, frac=FRAC)
    vec = timing_space(**DAY).vector()  # the identity timing = the baseline
    guarded = run_day(vector=vec, car_seed=1, abort=guard, **DAY)
    assert not guarded["aborted"]
    unguarded = run_day(vector=vec, car_seed=1, abort=RECORD_ONLY, **DAY)
    assert comparable(guarded) == comparable(unguarded)  # guard is inert


def test_guard_discards_gridlock_and_is_right_to():
    base = _base()
    guard = AbortGuard.from_baselines(base, frac=FRAC)
    gridlock = _gridlock_vector(timing_space(**DAY))

    guarded = run_day(vector=gridlock, car_seed=1, abort=guard, **DAY)
    assert guarded["aborted"]                       # discarded early...
    assert guarded["steps"] <= base[1]["steps"] * 0.55  # ...at half the cost
    assert day_score(guarded, base[1]) is None      # and scored as rejected

    # Ground truth: simulate the discarded day to the end — the collapse the
    # early check predicted really happens, so the discard was *right*.
    full = run_day(vector=gridlock, car_seed=1, **DAY)
    assert not full["aborted"]
    assert full["trips_completed"] < 0.5 * base[1]["trips_completed"]
    baseline_J = day_score(base[1], base[1])
    full_J = day_score(full, base[1])
    assert full_J is None or full_J < baseline_J


def test_guard_threshold_direction():
    base = _base()
    trips = base[1]["trips_at_frac"]
    # ratio 0 never aborts; an impossible ratio (>1) aborts even the baseline
    assert not AbortGuard({1: trips}, min_ratio=0.0).should_abort(1, 0)
    assert AbortGuard({1: trips}, min_ratio=1.1).should_abort(1, trips)
    # unknown seed (empty baseline) never aborts — the record-only guard
    assert not AbortGuard().should_abort(1, 0)


def test_evaluate_threads_the_guard_through():
    base = _base()
    guard = AbortGuard.from_baselines(base, frac=FRAC)
    gridlock = _gridlock_vector(timing_space(**DAY))
    ev = evaluate(gridlock, [1], base, workers=1, abort=guard, **DAY)
    assert ev.rejected and ev.J is None
    assert ev.days[0]["aborted"]
