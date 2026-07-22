"""Shared helpers for experiments.

:func:`build_default` assembles the **default city** — exactly the world
``python main.py`` runs (20x20 heterogeneous grid, elevated ring + expressway,
roundabouts, land use, activity schedules, 1000-car mixed fleet) — with keyword
overrides for any CLI flag (``cars=200``, ``day_length=300.0``, ``grade=False``,
...), so an experiment measures the same city the demo shows rather than a
private toy network.

:func:`pmap` fans a list of argument tuples over worker processes: experiments
sweep many independent seeded days and the simulator is single-threaded, so a
sweep parallelises perfectly.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Sequence, Tuple

from main import build_parser, build_simulation


def build_default(**overrides):
    """Build one default-city run: ``(net, sim, zones, args)``.

    ``overrides`` are CLI-flag attribute names (``cars``, ``car_seed``,
    ``day_length``, ``grade``, ...); an unknown name raises rather than being
    silently ignored. ``args.steps`` is resolved to one full day when not given,
    mirroring :func:`main.main`.
    """
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise AttributeError(f"unknown simulation option {k!r}")
        setattr(args, k, v)
    if args.steps is None:
        args.steps = max(1, round(args.day_length / args.dt))
    net, sim, zones = build_simulation(args)
    return net, sim, zones, args


def pmap(fn, jobs: Sequence[Tuple], workers: Optional[int] = None):
    """``[fn(*job) for job in jobs]``, over ``workers`` processes.

    ``workers=None`` uses one process per CPU (capped at the job count);
    ``workers=1`` runs in-process (handy under pytest). Results keep job order.
    """
    if workers == 1 or len(jobs) <= 1:
        return [fn(*job) for job in jobs]
    workers = min(len(jobs), workers or os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, *zip(*jobs)))
