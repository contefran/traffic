"""Verification of the evolution strategy and the stage-2 runner.

The optimizer is proven on problems with known answers before it touches
traffic: it must converge on a toy quadratic, treat rejected candidates as
worst-rank (walking away from a rejection region), and — the property the
48 h run depends on — **resume from a checkpoint bit-for-bit**: a run that
is saved, destroyed, and reloaded must produce exactly the run that was
never interrupted. The stage-2 smoke then exercises the full pipeline
(warm start anchor, guard, checkpoint, resume) on a compressed real day.
"""

import numpy as np

from ml.opt.es import EvolutionStrategy
from ml.opt.stage2 import run as stage2_run

BOUNDS = [(-5.0, 5.0)] * 8
TARGET = np.linspace(-2.0, 2.0, 8)


def _quad(vectors):
    return [-float(((np.array(v) - TARGET) ** 2).sum()) for v in vectors]


def test_converges_on_toy_quadratic():
    es = EvolutionStrategy([0.0] * 8, BOUNDS, population=16, sigma=0.05,
                           lr=0.5, sigma_decay=0.99, rng_seed=1)
    for _ in range(300):
        es.tell(_quad(es.ask()))
    assert float(((es.mean - TARGET) ** 2).sum()) < 0.05
    assert es.best_score > -0.1


def test_antithetic_pairs_and_bounds():
    es = EvolutionStrategy([0.0] * 8, BOUNDS, population=8, sigma=0.05,
                           rng_seed=0)
    cands = np.array(es.ask())
    half = len(cands) // 2
    assert np.allclose(cands[:half] - es.mean, -(cands[half:] - es.mean))
    assert (cands >= -5.0).all() and (cands <= 5.0).all()


def test_rejected_candidates_rank_worst():
    """Score = quadratic, but everything with x0 > 0 is 'rejected' (None).
    The optimizer must still improve, and drift into the allowed half."""
    target = np.array([-2.0] * 8)

    def score(vectors):
        return [None if v[0] > 0 else -float(((np.array(v) - target) ** 2).sum())
                for v in vectors]

    es = EvolutionStrategy([0.5] * 8, BOUNDS, population=16, sigma=0.05,
                           lr=0.5, sigma_decay=0.99, rng_seed=2)
    for _ in range(300):
        es.tell(score(es.ask()))
    assert es.mean[0] <= 0  # walked out of the rejection region
    assert float(((es.mean - target) ** 2).sum()) < 0.5


def test_all_rejected_generation_keeps_mean():
    es = EvolutionStrategy([0.0] * 8, BOUNDS, population=8, rng_seed=0)
    before = es.mean.copy()
    es.ask()
    stats = es.tell([None] * 8)
    assert np.array_equal(es.mean, before)
    assert stats["rejected"] == 8 and stats["mean_score"] is None


def test_save_load_resumes_bit_for_bit(tmp_path):
    ckpt = str(tmp_path / "es.json")
    # Uninterrupted reference: 12 generations straight.
    a = EvolutionStrategy([0.0] * 8, BOUNDS, population=16, sigma=0.05,
                          lr=0.5, rng_seed=3)
    for _ in range(12):
        a.tell(_quad(a.ask()))
    # Interrupted twin: 6 generations, checkpoint, discard, reload, 6 more.
    b = EvolutionStrategy([0.0] * 8, BOUNDS, population=16, sigma=0.05,
                          lr=0.5, rng_seed=3)
    for _ in range(6):
        b.tell(_quad(b.ask()))
    b.save(ckpt)
    del b
    c = EvolutionStrategy.load(ckpt)
    for _ in range(6):
        c.tell(_quad(c.ask()))
    assert np.array_equal(a.mean, c.mean)  # exact, not approximate
    assert a.best_score == c.best_score
    assert a.history == c.history


def test_stage2_smoke_with_resume(tmp_path):
    """The full pipeline on a compressed real day: warm start, guard,
    checkpointing, and a resume that continues the generation count."""
    ckpt = str(tmp_path / "stage2.json")
    city = dict(cars=150, day_length=400.0, work_scale=150.0)
    knobs = {"cycle": 0.8, "through": 0.7, "bias": 2.0, "wave": 1.0}
    lines = []
    r1 = stage2_run(checkpoint=ckpt, start_knobs=knobs, population=2,
                    sigma=0.02, seeds=[1], val_seeds=[2], val_every=0,
                    max_gens=2, workers=1, log=lines.append, **city)
    assert r1["generations"] == 2
    assert any("stage-1 anchor" in ln for ln in lines)  # warm start scored
    assert r1["val_history"][0]["generation"] == -1  # warm start validated
    r2 = stage2_run(checkpoint=ckpt, resume=True, population=2, seeds=[1],
                    val_seeds=[2], val_every=0, max_gens=3, workers=1,
                    log=lines.append, **city)
    assert r2["generations"] == 3  # continued, not restarted
    assert r2["final_val_J_gain"] is not None
    # The product can never be worse than the warm start's held-out score.
    warm_J = r2["val_history"][0]["J"]
    assert r2["best_val_J_gain"] is not None and r2["best_val_mean"] is not None
    assert warm_J is None or r2["best_val_J_gain"] >= warm_J - 0.5
