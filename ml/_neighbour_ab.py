"""A/B: does adding upstream/downstream neighbour features pay rent?

Fits the promoted scikit-learn trees with neighbours off vs on, same days,
same everything else, and prints the held-out MAE side by side — the
reproducible source of the neighbour-features table in the ML manual
(Step 6, measured August 2026).

Run (takes several minutes — it fits eight tree ensembles)::

    MPLBACKEND=Agg bin/python -m ml._neighbour_ab ml/data/varied
"""
import sys

import numpy as np

from ml.baselines import (evaluate_forecast, evaluate_occupancy, load_runs,
                          split_runs)
from ml.sklearn_models import SKTreesFlowModel

DATA = sys.argv[1] if len(sys.argv) > 1 else "ml/data/varied"
CELLS = [("counts", 60.0), ("counts", 10.0), ("speed", 10.0), ("speed", 60.0)]

runs, man = load_runs(DATA)
rest, test = split_runs(runs, (5,))
train, val = split_runs(rest, (4,))

print(f"{'cell':<16}{'off':>10}{'on':>10}{'delta':>10}{'trees off/on':>16}")
for target, hs in CELLS:
    h = max(1, round(hs / man["bin_s"]))
    key = "mae_cars" if target == "counts" else "mae_kmh"

    def score(neighbours):
        m = SKTreesFlowModel(h, man["day_length"], target=target,
                             neighbours=neighbours).fit(train, val)
        vals = []
        for day in test:
            pred = m.predict_day(day)
            if target == "speed":
                r = evaluate_forecast(pred, day["speed"], day["counts"],
                                      day["edge_speed_limit"], h)
            else:
                r = evaluate_occupancy(pred, day["counts"], h)
            vals.append(r[key])
        return float(np.mean(vals)), m.n_trees_

    off, n_off = score(False)
    on, n_on = score(True)
    unit = "cars" if target == "counts" else "km/h"
    print(f"{target+'@'+str(int(hs))+'s':<16}{off:>10.4f}{on:>10.4f}"
          f"{on - off:>+10.4f}{f'{n_off}/{n_on}':>16}  {unit}")
