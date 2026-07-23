"""Generate the flow-model training dataset: simulated days -> per-edge series.

The flow-prediction model (the portfolio's supervised-learning step) learns to
predict near-future traffic conditions on every street from the recent past.
Its raw material is exactly what ``MetricsCollector(record_edges=True)``
observes during a simulated day: per step and per occupied edge, the vehicle
count and mean speed. This module runs a sweep of seeded full days at varying
fleet sizes (load levels), aggregates the 0.1 s-step series into coarser time
bins, and writes one compressed ``.npz`` per run plus a ``manifest.json``
describing the whole sweep.

Design decisions (the dataset's contract):

* **The network is fixed across runs** (same builder arguments and seed): the
  model learns *one city's* dynamics, and variation comes from demand
  (``car_seed``) and load (``cars``). Consequently column ``e`` of every array
  is the same street in every file, and static edge features are stored once
  per run.
* **Binning.** ``dt = 0.1 s`` snapshots are far finer than the phenomena the
  model cares about; averaging into ``bin_s``-second bins (default 10 s)
  shrinks the data ~100x and denoises it. ``counts[b, e]`` is the time-mean
  vehicle count on edge ``e`` during bin ``b`` (occupancy); ``speed[b, e]`` is
  the vehicle-weighted mean speed [m/s], ``NaN`` where nothing drove.
* **Parked (inactive) cars are excluded** by construction — the collector only
  sees active cars — so a residential street at night reads count 0 / speed
  ``NaN``, which is what a road sensor would say.

Run (from the repo root)::

    MPLBACKEND=Agg bin/python -m ml.dataset --out ml/data/default
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from experiments.common import build_default, pmap
from main import build_parser
from traffic_sim.zones import LandUse

# Stable integer codes for the land-use zone of each edge (-1 = unzoned,
# e.g. elevated highway or roundabout ring).
ZONE_CODES = {use: k for k, use in enumerate(LandUse)}


def run_day(cars: int, car_seed: int, bin_s: float = 10.0, **overrides) -> dict:
    """Simulate one full day and return its binned per-edge arrays.

    ``overrides`` are forwarded to :func:`experiments.common.build_default`
    (any main.py CLI flag), so tests can shrink the city/day. Returns a dict of
    numpy arrays — the exact payload :func:`generate` saves per run — plus a
    few scalar sanity stats (``crashes``, ``trips_completed``).
    """
    net, sim, zones, args = build_default(cars=cars, car_seed=car_seed,
                                          **overrides)
    sim.metrics.record_edges = True
    for _ in range(args.steps):
        sim.step(args.dt)

    n_edges = len(net.edges)
    bin_steps = max(1, round(bin_s / args.dt))
    n_bins = math.ceil(args.steps / bin_steps)
    # Accumulate sum-of-counts and sum-of(count * speed) per (bin, edge), then
    # normalize: time-mean occupancy and vehicle-weighted mean speed.
    count_sum = np.zeros((n_bins, n_edges))
    speed_sum = np.zeros((n_bins, n_edges))
    for k, snap in enumerate(sim.metrics.edge_history):
        b = k // bin_steps
        for eid, (cnt, v) in snap.items():
            count_sum[b, eid] += cnt
            speed_sum[b, eid] += cnt * v
    steps_per_bin = np.full(n_bins, bin_steps, dtype=float)
    steps_per_bin[-1] = args.steps - (n_bins - 1) * bin_steps  # last, partial bin
    counts = count_sum / steps_per_bin[:, None]
    with np.errstate(invalid="ignore"):
        speed = np.where(count_sum > 0.0, speed_sum / np.maximum(count_sum, 1e-12),
                         np.nan)

    summary = sim.metrics.summary()
    return {
        # Time-varying (the model's inputs *and* targets).
        "counts": counts.astype(np.float32),
        "speed": speed.astype(np.float32),
        "bin_t": (np.arange(n_bins) * bin_steps * args.dt).astype(np.float32),
        # Static edge features (column e = edge id e, identical across runs).
        "edge_length": np.array([e.length for e in net.edges], np.float32),
        "edge_speed_limit": np.array([e.speed_limit for e in net.edges],
                                     np.float32),
        "edge_lanes": np.array([e.lanes for e in net.edges], np.int16),
        "edge_level": np.array([net.nodes[e.u].level for e in net.edges],
                               np.int16),
        "edge_zone": np.array([ZONE_CODES.get(zones.get(e.id), -1)
                               for e in net.edges], np.int16),
        # Topology (u -> v node ids), so a model can later use neighbours.
        "edge_u": np.array([e.u for e in net.edges], np.int32),
        "edge_v": np.array([e.v for e in net.edges], np.int32),
        # Scalar sanity stats, surfaced into the manifest.
        "crashes": summary.get("crashes", 0),
        "trips_completed": summary.get("trips_completed", 0),
    }


def _write_run(out_dir: str, cars: int, car_seed: int, bin_s: float,
               intensity: float, overrides: dict) -> dict:
    """Worker: simulate one day, save its ``.npz``, return a manifest row."""
    data = run_day(cars, car_seed, bin_s, intensity=intensity, **overrides)
    crashes = int(data.pop("crashes"))
    trips = int(data.pop("trips_completed"))
    name = f"run_cars{cars}_seed{car_seed}.npz"
    np.savez_compressed(Path(out_dir) / name, **data)
    return {"file": name, "cars": cars, "car_seed": car_seed,
            "intensity": intensity, "crashes": crashes,
            "trips_completed": trips}


def _day_intensity(cars: int, car_seed: int, lo: float, hi: float) -> float:
    """The day's demand intensity: uniform in ``[lo, hi]``, seeded per run.

    This is the *unpredictable-but-observable* day-to-day variation (busy
    Mondays, quiet Sundays) that makes medium-horizon forecasting a real
    problem. It is recorded in the manifest for analysis, but a model must
    never receive it as a feature — in reality you learn how busy today is by
    watching the streets, which is exactly what the model should do.
    """
    return random.Random(car_seed * 100003 + cars).uniform(lo, hi)


def generate(out_dir, loads=(300, 600, 1000, 1400), seeds=(1, 2, 3),
             bin_s: float = 10.0, workers=None,
             intensity_range=(1.0, 1.0), **overrides) -> dict:
    """Run the ``loads x seeds`` sweep and write the dataset to ``out_dir``.

    One ``.npz`` per (load, seed) day plus a ``manifest.json`` recording the
    sweep configuration and per-run sanity stats. ``intensity_range=(lo, hi)``
    draws each day's demand intensity uniformly (seeded per run); the default
    ``(1.0, 1.0)`` keeps every day at full, identical demand. Runs fan out
    over processes via :func:`experiments.common.pmap` (``workers=1`` runs
    in-process, handy under pytest). Returns the manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = intensity_range
    jobs = [(str(out), cars, seed, bin_s, _day_intensity(cars, seed, lo, hi),
             overrides)
            for cars in loads for seed in seeds]
    rows = pmap(_write_run, jobs, workers)

    # Record the effective day/step config alongside the sweep definition.
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    manifest = {
        "bin_s": bin_s,
        "dt": args.dt,
        "day_length": args.day_length,
        "loads": list(loads),
        "seeds": list(seeds),
        "intensity_range": list(intensity_range),
        "overrides": overrides,
        "zone_codes": {use.name: k for use, k in ZONE_CODES.items()},
        "runs": rows,
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv=None) -> None:
    """CLI wrapper around :func:`generate`."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="ml/data/default",
                   help="output directory for the .npz runs + manifest.json")
    p.add_argument("--cars", type=int, nargs="+", default=[300, 600, 1000, 1400],
                   help="fleet sizes (load levels) to sweep")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="car-placement seeds per load (demand realizations)")
    p.add_argument("--bin", type=float, default=10.0, dest="bin_s",
                   help="aggregation bin [s] for the per-edge series")
    p.add_argument("--workers", type=int, default=None,
                   help="worker processes (default: one per CPU)")
    p.add_argument("--intensity-range", type=float, nargs=2, default=(1.0, 1.0),
                   metavar=("LO", "HI"),
                   help="per-day demand intensity drawn uniformly from "
                        "[LO, HI] (seeded per run); 1 1 = identical full-"
                        "demand days")
    args = p.parse_args(argv)
    manifest = generate(args.out, loads=args.cars, seeds=args.seeds,
                        bin_s=args.bin_s, workers=args.workers,
                        intensity_range=tuple(args.intensity_range))
    for row in manifest["runs"]:
        print(f"  {row['file']}: intensity {row['intensity']:.2f}, "
              f"{row['trips_completed']} trips, {row['crashes']} crashes")
    print(f"wrote {len(manifest['runs'])} runs to {args.out}")


if __name__ == "__main__":
    main()
