"""From-scratch evolution strategy (OpenAI-ES style) with checkpointing.

The optimizer for stage 2 (``guides_plans/rl_plan.md`` §6): a simple,
readable **evolution strategy** — each generation samples a population of
Gaussian perturbations around the current mean vector, scores them all, and
moves the mean along the score-weighted average direction. Design choices:

* **Antithetic pairs**: perturbations come in ``+eps / -eps`` pairs, so the
  estimated direction is unbiased even with a small population.
* **Rank shaping**: candidates are scored by their *rank* in the generation
  (centered to [-0.5, 0.5]), not their raw J — robust to the objective's
  scale and to outliers. Rejected candidates (score ``None``: the crash cap
  or the gridlock guard fired) simply get the worst ranks, so the search
  walks away from catastrophe without any special-case magic.
* **Per-dimension scale**: one global step size ``sigma``, expressed as a
  fraction of each dimension's box range (greens span 57 s, offsets 240 s —
  a single absolute step would be meaningless), applied both to sampling
  and to the update.
* **Checkpointing**: :meth:`state`/:meth:`save`/:meth:`load` serialize the
  *complete* optimizer — mean, best-so-far, generation counter, history and
  the RNG state — as JSON, written atomically. A loaded optimizer continues
  the exact run: same RNG stream, same candidates, same updates
  (unit-tested; this is what makes a 48 h run killable at any point).

The optimizer is deliberately decoupled from traffic: it sees only vectors,
bounds, and a batch of scores. ``ml/opt/stage2.py`` supplies the traffic.
"""

import json
import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


class EvolutionStrategy:
    """Rank-shaped antithetic ES over a box-bounded vector.

    ``wrap`` (optional) canonicalizes a vector after sampling and after the
    mean update — stage 2 passes offset-wrapping, which is behaviour-
    preserving (plans are periodic) but keeps values inside bounds.
    """

    def __init__(self, start: Sequence[float],
                 bounds: Sequence[Tuple[float, float]], *,
                 population: int = 32, sigma: float = 0.02, lr: float = 0.2,
                 sigma_decay: float = 1.0, rng_seed: int = 0,
                 wrap: Optional[Callable[[List[float]], List[float]]] = None):
        """``sigma_decay`` < 1 anneals the search: it geometrically shrinks
        **both** ``sigma`` (the sampling radius) and ``lr`` each generation.
        Both, because under rank shaping the update magnitude is set by
        ``lr`` alone — ranks have the same scale at any sampling radius —
        so decaying sigma by itself would shrink exploration while the mean
        keeps jittering at full step size. At 1.0 (default) the search
        never settles below a step-sized radius — acceptable for a
        fine-tune, wrong for exact convergence."""
        if population % 2:
            raise ValueError("population must be even (antithetic pairs)")
        self.lo = np.array([b[0] for b in bounds], float)
        self.hi = np.array([b[1] for b in bounds], float)
        self.scale = self.hi - self.lo
        self.population = population
        self.sigma = sigma
        self.lr = lr
        self.sigma_decay = sigma_decay
        self.wrap = wrap
        self.rng = np.random.default_rng(rng_seed)
        self.mean = self._legal(np.asarray(start, float))
        self.generation = 0
        self.best: Optional[List[float]] = None
        self.best_score = -np.inf
        self.history: List[dict] = []
        self._eps: Optional[np.ndarray] = None
        self._cands: Optional[np.ndarray] = None

    def _legal(self, vec: np.ndarray) -> np.ndarray:
        if self.wrap is not None:
            vec = np.asarray(self.wrap(list(vec)), float)
        return np.clip(vec, self.lo, self.hi)

    # ------------------------------------------------------------ ask/tell

    def ask(self) -> List[List[float]]:
        """The next generation's candidate vectors (antithetic, clipped)."""
        half = self.population // 2
        eps = self.rng.standard_normal((half, self.mean.size))
        self._eps = np.concatenate([eps, -eps])
        cands = self.mean + self.sigma * self.scale * self._eps
        self._cands = np.array([self._legal(c) for c in cands])
        return [list(c) for c in self._cands]

    def tell(self, scores: Sequence[Optional[float]]) -> dict:
        """Consume the candidates' scores; update mean, best, and history.

        ``None`` scores (rejected candidates) rank below every real score.
        If the *whole* generation was rejected the mean stays put — a blind
        step on meaningless ranks would be noise, not progress.
        """
        if self._eps is None or len(scores) != self.population:
            raise ValueError("tell() must follow ask() with matching scores")
        raw = np.array([-np.inf if s is None else s for s in scores])
        scored = np.isfinite(raw)
        if scored.any():
            i = int(raw.argmax())
            if raw[i] > self.best_score:
                self.best_score = float(raw[i])
                self.best = list(self._cands[i])
            ranks = raw.argsort().argsort()  # -inf sorts first = worst rank
            util = ranks / (self.population - 1) - 0.5
            grad = (util[:, None] * self._eps).mean(axis=0)
            self.mean = self._legal(self.mean + self.lr * self.scale * grad)
        stats = {
            "generation": self.generation,
            "mean_score": (float(raw[scored].mean()) if scored.any() else None),
            "best_in_gen": (float(raw.max()) if scored.any() else None),
            "best_so_far": (None if self.best is None else self.best_score),
            "rejected": int((~scored).sum()),
        }
        self.generation += 1
        self.sigma *= self.sigma_decay
        self.lr *= self.sigma_decay  # see __init__: ranks don't scale with sigma
        self.history.append(stats)
        self._eps = self._cands = None
        return stats

    # --------------------------------------------------------- persistence

    def state(self) -> dict:
        """The complete optimizer as JSON-serializable data."""
        return {
            "mean": list(self.mean), "lo": list(self.lo), "hi": list(self.hi),
            "population": self.population, "sigma": self.sigma, "lr": self.lr,
            "sigma_decay": self.sigma_decay,
            "generation": self.generation,
            "best": self.best,
            "best_score": (None if self.best is None else self.best_score),
            "history": self.history,
            "rng_state": self.rng.bit_generator.state,
        }

    def save(self, path: str) -> None:
        """Atomic checkpoint write (never a half-written file)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state(), fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str,
             wrap: Optional[Callable[[List[float]], List[float]]] = None
             ) -> "EvolutionStrategy":
        """Rebuild an optimizer mid-run; continues the exact RNG stream."""
        with open(path) as fh:
            s = json.load(fh)
        es = cls(s["mean"], list(zip(s["lo"], s["hi"])),
                 population=s["population"], sigma=s["sigma"], lr=s["lr"],
                 sigma_decay=s.get("sigma_decay", 1.0), wrap=wrap)
        es.generation = s["generation"]
        es.best = s["best"]
        es.best_score = (-np.inf if s["best_score"] is None
                         else s["best_score"])
        es.history = s["history"]
        es.rng.bit_generator.state = s["rng_state"]
        return es
