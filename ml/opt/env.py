"""Evaluation harness for signal-timing optimization: vector in, score out.

This module is the bridge between an optimizer and the simulator. A candidate
**timing vector** — every signalized intersection's green times and offset,
flattened into one list by :class:`traffic_sim.tuning.ParameterSpace` — goes
in; full seeded days of the default city are simulated under it; a scalar
score comes out. Everything an optimizer (grid sweep, evolution strategy)
needs lives here, so the optimizers themselves stay algorithm-only.

Design decisions (full rationale in ``guides_plans/rl_plan.md``):

* **Common random numbers.** Candidates are only ever compared on the same
  fixed set of demand seeds, because day-to-day noise (measured: sd ~56
  trips, ~1.8 s mean delay across seeds) is as large as the effects being
  chased. The simulator is deterministic per seed, so a same-seed difference
  between two candidates is exact, not sampled.
* **The score** for one day is measured against the *same-seed baseline* day
  run on default plans: ``J = trips/trips_base - w * delay/delay_base``.
  Both terms are ratios near 1.0, so ``w`` reads as an exchange rate
  ("w = 0.5: a 2% delay cut is worth a 1% throughput loss"). Trips completed
  carries real weight because mean delay alone is survivorship-biased at
  saturation — when the network jams, only the lucky trips finish and their
  mean delay looks *good* (the q18 lesson).
* **Crashes are a guardrail, not an objective.** A day exceeding
  ``crash_cap`` rejects the candidate outright (score ``None``) instead of
  trading off against flow — we do not want an optimizer discovering that
  gridlock prevents collisions.
* **Offsets are periodic.** :func:`wrap_offsets` reduces each node's offset
  modulo its own cycle, which never changes behaviour
  (``SignalPlan.phase_state`` is ``(t - offset) % cycle``) but keeps vectors
  inside ``ParameterSpace`` bounds. Wrap before ``apply()`` clips: a large
  legitimate offset (e.g. ``apply_green_wave`` installs raw distance/speed
  offsets up to ~300 s on this city, past the 240 s box bound) would
  otherwise be corrupted by clipping.
"""

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

from experiments.common import build_default, pmap
from traffic_sim.tuning import ParameterSpace

#: Default exchange rate between the throughput and delay terms of the score.
DEFAULT_W = 0.5
#: Reject any candidate with a day above this many crashes (baseline: 0-3).
DEFAULT_CRASH_CAP = 5
#: Fraction of the day at which every run records its trips-so-far (and an
#: :class:`AbortGuard`, if given, takes its abort decision). 0.4 = ~09:36:
#: the checkpoint must sit *after* the morning rush has completed trips —
#: the rush starts at exactly 0.25 (06:00) and a commute takes ~80 s of
#: physical (non-compressible) travel, so an earlier checkpoint counts only
#: night traffic and cannot tell a jammed rush from a healthy one.
ABORT_FRAC = 0.4
#: Abort when trips-so-far fall below this ratio of the same-seed baseline's.
DEFAULT_ABORT_RATIO = 0.4


@dataclass
class AbortGuard:
    """Deterministic gridlock early-abort: stop simulating hopeless days.

    A day's simulation cost is *endogenous* — a terrible timing jams the
    city, keeps every car active, and costs several times a baseline day to
    simulate (measured in the stage-1 sweep). The guard bounds that tail: at
    ``frac`` of the day it compares trips completed so far against the
    same-seed baseline's count at that same point (``baseline`` maps seed →
    trips), and if the ratio is below ``min_ratio`` the day is aborted and
    scored as rejected — it was headed for a terrible J anyway, so the
    optimizer loses nothing while the machine gets ~three quarters of the
    day back. The test is in-sim and deterministic, never wall-clock.

    A guard with an empty ``baseline`` never aborts (unknown seeds pass) —
    useful to make :func:`run_day` *record* ``trips_at_frac`` at a custom
    ``frac`` when measuring baselines for that frac.
    """

    baseline: Dict[int, int] = field(default_factory=dict)
    min_ratio: float = DEFAULT_ABORT_RATIO
    frac: float = ABORT_FRAC

    def should_abort(self, seed: int, trips_so_far: int) -> bool:
        base = self.baseline.get(seed)
        return base is not None and trips_so_far < self.min_ratio * base

    @classmethod
    def from_baselines(cls, baselines: Dict[int, dict],
                       min_ratio: float = DEFAULT_ABORT_RATIO,
                       frac: float = ABORT_FRAC) -> "AbortGuard":
        """Build the guard from :func:`measure_baselines` output.

        ``frac`` must match the fraction the baselines recorded
        ``trips_at_frac`` at — the default :data:`ABORT_FRAC` unless the
        baselines were measured with a record-only guard
        (``abort=AbortGuard(frac=...)``) at a custom fraction."""
        return cls({s: d["trips_at_frac"] for s, d in baselines.items()},
                   min_ratio, frac)


# --------------------------------------------------------------- one day

def run_day(vector: Optional[Sequence[float]] = None, car_seed: int = 1,
            abort: Optional[AbortGuard] = None, **overrides) -> dict:
    """Simulate one full seeded day, optionally under a candidate vector.

    Builds the default city (any ``build_default`` override accepted, e.g.
    ``cars=``, ``green_wave=True``), applies ``vector`` to its signal plans
    via :class:`ParameterSpace` (``None`` = leave the built plans untouched),
    runs the whole day, and returns the metrics summary dict, extended with:

    * ``trips_at_frac`` — trips completed at the checkpoint fraction of the
      day (recorded on every run; the yardstick :class:`AbortGuard` reads),
    * ``aborted`` — True when ``abort`` cut the day short at the checkpoint
      (the summary then covers only the simulated part),
    * ``wall_s`` — wall-clock cost of the run (calibration data; strip it
      with :func:`comparable` before determinism comparisons).
    """
    t0 = time.perf_counter()
    net, sim, zones, args = build_default(car_seed=car_seed, **overrides)
    if vector is not None:
        ParameterSpace(sim.signals).apply(vector)
    frac = abort.frac if abort is not None else ABORT_FRAC
    check_step = max(1, round(frac * args.steps))
    trips_at_frac = None
    aborted = False
    for i in range(args.steps):
        sim.step(args.dt)
        if i + 1 == check_step:
            trips_at_frac = len(sim.metrics.trips)
            if abort is not None and abort.should_abort(car_seed,
                                                        trips_at_frac):
                aborted = True
                break
    m = sim.metrics.summary()
    m["trips_at_frac"] = trips_at_frac
    m["aborted"] = aborted
    m["wall_s"] = time.perf_counter() - t0
    return m


def comparable(day: dict) -> dict:
    """``day`` without its wall-clock field — for bit-identity comparisons
    (the dynamics are deterministic; the wall clock never is)."""
    return {k: v for k, v in day.items() if k != "wall_s"}


# ---------------------------------------------------------------- vectors

def timing_space(**overrides) -> ParameterSpace:
    """A :class:`ParameterSpace` over a fresh default city's signals.

    The network is identical across ``car_seed`` (and ``cars``) by
    construction, so this layout matches every day :func:`run_day` builds:
    a vector produced against this space applies to any of them.
    """
    net, sim, zones, args = build_default(**overrides)
    return ParameterSpace(sim.signals)


def wrap_offsets(space: ParameterSpace, vector: Sequence[float]) -> List[float]:
    """A copy of ``vector`` with each node's offset reduced modulo its cycle.

    Behaviour-preserving (plans are periodic in the offset) and keeps the
    offset dimensions inside ``space.bounds()`` so ``apply()``'s clipping
    never bites. The cycle is computed from the vector's *own* green times
    plus the node's (fixed, safety-ruled) yellow.
    """
    vec = list(vector)
    if not space.include_offsets:
        return vec
    per = space._per_node
    for k, node in enumerate(space.nodes):
        i = k * per
        yellow = space.controller.plan_for(node).yellow
        cycle = sum(vec[i:i + space.num_phases]) + space.num_phases * yellow
        if cycle > 0:
            vec[i + space.num_phases] %= cycle
    return vec


def green_wave_vector(**overrides) -> List[float]:
    """The classical green-wave baseline, encoded as a timing vector.

    Builds the city with ``--green-wave`` applied and reads the resulting
    plans back through :class:`ParameterSpace` (offsets wrapped). Running
    :func:`run_day` under this vector is bit-for-bit the ``--green-wave``
    city — the harness's second regression anchor, and stage 1's warm start.
    """
    net, sim, zones, args = build_default(green_wave=True, **overrides)
    space = ParameterSpace(sim.signals)
    return wrap_offsets(space, space.vector())


# ---------------------------------------------------------------- scoring

def day_score(day: dict, base: dict, w: float = DEFAULT_W,
              crash_cap: int = DEFAULT_CRASH_CAP) -> Optional[float]:
    """Score one day against its same-seed baseline; ``None`` = rejected.

    ``J = trips/trips_base - w * delay/delay_base``; the baseline day itself
    scores exactly ``1 - w``, so anything above that is an improvement.
    Aborted days (the gridlock guard fired) are rejected outright.
    """
    if day.get("aborted"):
        return None
    if day["crashes"] > crash_cap:
        return None
    return (day["trips_completed"] / base["trips_completed"]
            - w * day["mean_delay_s"] / base["mean_delay_s"])


@dataclass
class Evaluation:
    """One candidate's measured days and per-seed scores, in seed order."""

    seeds: List[int]
    days: List[dict]
    scores: List[Optional[float]]

    @property
    def rejected(self) -> bool:
        """True when any day tripped the crash cap."""
        return any(s is None for s in self.scores)

    @property
    def J(self) -> Optional[float]:
        """Mean score over the seeds (``None`` when rejected)."""
        if self.rejected:
            return None
        return sum(self.scores) / len(self.scores)


def measure_baselines(seeds: Sequence[int], workers: Optional[int] = None,
                      **overrides) -> Dict[int, dict]:
    """The default-plan day for each seed: the fixed yardstick J is scored
    against. Measure once per protocol and reuse for every candidate."""
    days = pmap(partial(run_day, **overrides),
                [(None, s) for s in seeds], workers=workers)
    return dict(zip(seeds, days))


def evaluate(vector: Optional[Sequence[float]], seeds: Sequence[int],
             baselines: Dict[int, dict], w: float = DEFAULT_W,
             crash_cap: int = DEFAULT_CRASH_CAP,
             workers: Optional[int] = None,
             abort: Optional[AbortGuard] = None, **overrides) -> Evaluation:
    """Run ``vector`` over ``seeds`` (in parallel) and score each day.

    ``baselines`` comes from :func:`measure_baselines` on the same seeds and
    overrides — same-seed comparison is what makes scores exact under
    common random numbers. Pass ``abort=AbortGuard.from_baselines(baselines)``
    to cut hopeless (gridlocked) days short instead of simulating them out.
    """
    days = pmap(partial(run_day, abort=abort, **overrides),
                [(vector, s) for s in seeds], workers=workers)
    scores = [day_score(d, baselines[s], w, crash_cap)
              for s, d in zip(seeds, days)]
    return Evaluation(seeds=list(seeds), days=days, scores=scores)
