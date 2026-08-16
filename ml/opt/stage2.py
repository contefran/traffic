"""Stage 2: full per-node ES fine-tune from the stage-1 warm start.

Wires :class:`ml.opt.es.EvolutionStrategy` to the traffic harness
(``guides_plans/rl_plan.md`` §6): the 1670-dimension timing vector, warm-
started from the stage-1 winner's knobs, scored on the training seeds under
common random numbers with the gridlock guard active, wall-clock-budgeted,
checkpointed every generation, and periodically validated on the held-out
validation seeds (early-stopping on a validation plateau).

Usage::

    MPLBACKEND=Agg bin/python -u -m ml.opt.stage2 --budget-hours 48
    MPLBACKEND=Agg bin/python -u -m ml.opt.stage2 --resume [--budget-hours H]

The checkpoint (``ml/data/opt/stage2_ckpt.json``) carries the complete
optimizer plus the validation history, so ``--resume`` continues the exact
run — same RNG stream, same candidates — after a kill, crash, or reboot.
Startup logs the warm start's train score, which must reproduce stage 1's
measured value exactly (same seeds, deterministic sim) — a free full-scale
anchor on every launch.
"""

import argparse
import json
import os
import time
from functools import partial
from typing import Dict, Optional, Sequence

from experiments.common import pmap
from ml.opt.baselines import TRAIN_SEEDS, VAL_SEEDS
from ml.opt.env import (DEFAULT_CRASH_CAP, DEFAULT_W, AbortGuard, day_score,
                        evaluate, measure_baselines, run_day, timing_space,
                        wrap_offsets)
from ml.opt.es import EvolutionStrategy
from ml.opt.stage1 import stage1_context, structured_vector

CKPT = "ml/data/opt/stage2_ckpt.json"
RESULT = "ml/data/opt/stage2_result.json"
STAGE1_BEST = "ml/data/opt/stage1_best.json"


def batch_score(vectors, seeds, baselines, guard, w=DEFAULT_W,
                crash_cap=DEFAULT_CRASH_CAP, workers=None, **overrides):
    """Mean J per vector over ``seeds`` — all days in one parallel fan-out.

    One flat job list (every candidate × every seed) keeps all workers busy
    regardless of population/seed shape. A vector whose any day is rejected
    (crash cap or gridlock guard) scores ``None``.
    """
    jobs = [(list(v), s) for v in vectors for s in seeds]
    days = pmap(partial(run_day, abort=guard, **overrides), jobs,
                workers=workers)
    out = []
    for i in range(len(vectors)):
        mine = days[i * len(seeds):(i + 1) * len(seeds)]
        scores = [day_score(d, baselines[s], w, crash_cap)
                  for s, d in zip(seeds, mine)]
        out.append(None if any(x is None for x in scores)
                   else sum(scores) / len(scores))
    return out


def run(*, checkpoint: str = CKPT, resume: bool = False,
        start_knobs: Optional[Dict[str, float]] = None,
        population: int = 128, sigma: float = 0.01, lr: float = 0.2,
        rng_seed: int = 0,
        seeds: Sequence[int] = TRAIN_SEEDS,
        val_seeds: Sequence[int] = VAL_SEEDS,
        val_every: int = 5, patience: int = 6,
        max_gens: Optional[int] = None, budget_s: Optional[float] = None,
        w: float = DEFAULT_W, crash_cap: int = DEFAULT_CRASH_CAP,
        workers: Optional[int] = None, log=print, **overrides) -> dict:
    """The stage-2 loop. Returns a result dict (also written to disk).

    ``start_knobs`` overrides the stage-1 winner (tests pass small knobs;
    the default reads ``ml/data/opt/stage1_best.json``). ``overrides`` are
    city-build settings (tests use a compressed day). ``val_every=0``
    disables mid-run validation (final validation always runs).
    """
    t0 = time.time()
    space, base_offsets, node_fast = stage1_context(**overrides)
    wrap = lambda v: wrap_offsets(space, v)

    log(f"stage2: measuring baselines on train {list(seeds)} "
        f"+ val {list(val_seeds)}")
    baselines = measure_baselines(seeds, workers=workers, **overrides)
    val_base = measure_baselines(val_seeds, workers=workers, **overrides)
    guard = AbortGuard.from_baselines(baselines)

    if resume:
        es = EvolutionStrategy.load(checkpoint, wrap=wrap)
        with open(checkpoint) as fh:
            aux = json.load(fh).get("stage2", {})
        val_history = aux.get("val_history", [])
        best_val = aux.get("best_val")
        best_val_mean = aux.get("best_val_mean")
        stale = aux.get("stale", 0)
        log(f"stage2: resumed at generation {es.generation}")
    else:
        if start_knobs is None:
            with open(STAGE1_BEST) as fh:
                start_knobs = json.load(fh)["knobs"]
        warm = structured_vector(space, base_offsets, node_fast, **start_knobs)
        es = EvolutionStrategy(warm, space.bounds(), population=population,
                               sigma=sigma, lr=lr, rng_seed=rng_seed,
                               wrap=wrap)
        val_history, stale = [], 0
        start_J = batch_score([warm], seeds, baselines, guard, w, crash_cap,
                              workers, **overrides)[0]
        log(f"stage2: warm start {start_knobs} -> train J-(1-w) "
            f"{start_J - (1 - w):+.4f} (stage-1 anchor)")
        # The warm start is validation's opening bar: the ES's mean must
        # beat it on held-out seeds or the delivered product is stage 1's.
        warm_val = evaluate(warm, val_seeds, val_base, w=w,
                            crash_cap=crash_cap, workers=workers, **overrides)
        best_val = None if warm_val.rejected else warm_val.J
        best_val_mean = list(warm)
        val_history.append({"generation": -1, "J": best_val})
        log(f"stage2: warm start validation J-(1-w) "
            + ("rejected" if best_val is None
               else f"{best_val - (1 - w):+.4f}") + " (the bar to beat)")

    def save():
        state = es.state()
        state["stage2"] = {"val_history": val_history, "best_val": best_val,
                           "best_val_mean": best_val_mean, "stale": stale}
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        tmp = checkpoint + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, checkpoint)

    while True:
        if max_gens is not None and es.generation >= max_gens:
            log("stage2: generation limit reached")
            break
        if budget_s is not None and time.time() - t0 >= budget_s:
            log("stage2: wall-clock budget reached")
            break
        gen_t = time.time()
        cands = es.ask()
        scores = batch_score(cands, seeds, baselines, guard, w, crash_cap,
                             workers, **overrides)
        stats = es.tell(scores)
        gen = stats["generation"]
        log(f"stage2: gen {gen}: mean "
            + ("all-rejected" if stats["mean_score"] is None
               else f"{stats['mean_score'] - (1 - w):+.4f}")
            + f", best-in-gen "
            + ("-" if stats["best_in_gen"] is None
               else f"{stats['best_in_gen'] - (1 - w):+.4f}")
            + f", rejected {stats['rejected']}/{len(cands)}, "
            f"{time.time() - gen_t:.0f}s, elapsed {(time.time()-t0)/3600:.2f}h")

        if val_every and (gen + 1) % val_every == 0:
            ev = evaluate(list(es.mean), val_seeds, val_base, w=w,
                          crash_cap=crash_cap, workers=workers, **overrides)
            vj = None if ev.rejected else ev.J
            val_history.append({"generation": gen, "J": vj})
            if vj is not None and (best_val is None or vj > best_val):
                best_val = vj
                best_val_mean = list(es.mean)
                stale = 0
            else:
                stale += 1
            log(f"stage2: validation at gen {gen}: "
                + ("rejected" if vj is None else f"J-(1-w) {vj - (1 - w):+.4f}")
                + f" (best {'-' if best_val is None else f'{best_val - (1-w):+.4f}'}, "
                f"stale {stale}/{patience})")
            if patience and stale >= patience:
                save()
                log("stage2: validation plateau — early stop")
                break
        save()

    final = evaluate(list(es.mean), val_seeds, val_base, w=w,
                     crash_cap=crash_cap, workers=workers, **overrides)
    final_J = None if final.rejected else final.J
    if final_J is not None and (best_val is None or final_J > best_val):
        best_val = final_J
        best_val_mean = list(es.mean)
    result = {
        "generations": es.generation,
        "elapsed_h": (time.time() - t0) / 3600,
        "train_best_J_gain": (None if es.best is None
                              else es.best_score - (1 - w)),
        "final_val_J_gain": (None if final_J is None else final_J - (1 - w)),
        # The product: the mean with the best held-out score ever seen —
        # initialized to the warm start, so a drifting search can never
        # deliver less than stage 1 did.
        "best_val_J_gain": (None if best_val is None else best_val - (1 - w)),
        "best_val_mean": best_val_mean,
        "val_history": val_history,
        "mean": list(es.mean),
    }
    result_path = RESULT if checkpoint == CKPT else checkpoint + ".result.json"
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w") as fh:
        json.dump(result, fh)
    log(f"stage2: done — {es.generation} generations, final validation "
        + ("rejected" if final.rejected
           else f"J-(1-w) {final.J - (1 - w):+.4f}"))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-hours", type=float, default=48.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-gens", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    run(resume=args.resume, population=args.population, sigma=args.sigma,
        lr=args.lr, val_every=args.val_every, patience=args.patience,
        max_gens=args.max_gens, budget_s=args.budget_hours * 3600,
        workers=args.workers)


if __name__ == "__main__":
    main()
