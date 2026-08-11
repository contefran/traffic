"""The flow model as a web service: FastAPI around a saved bundle.

The deliverable of the ML phase is not a table of MAEs — it is *this*: a
process that loads the fitted bundle (:mod:`ml.artifact`) once at startup
and answers forecast queries over HTTP in milliseconds. The contract:

* ``GET /health`` — the bundle's own scorecard: which (target, horizon)
  cells are loaded, their tree counts and measured test MAE, the network
  size, library versions.
* ``POST /predict`` — send *today's observations so far* (per-edge speed
  and occupancy for every bin since midnight, ``null`` speed = nobody
  drove) and a cell; receive the per-street forecast ``horizon_s`` ahead
  of the last observed bin.

Serving-skew note (the trap the manual warns about): this module never
builds a feature. The request is handed to the pickled model's
:meth:`~ml.linear.LinearFlowModel.predict_next`, which pads the history
and runs the *training* feature pipeline on it — the served forecast is
bit-identical to what the offline evaluation would compute at that bin.
Static street facts ride inside the bundle (the network is fixed — the
dataset contract), so clients only send what a road sensor would see.

Run (after ``bin/python -m ml.artifact``)::

    bin/python -m ml.serve                  # http://127.0.0.1:8000/docs
    bin/uvicorn ml.serve:create_app --factory   # equivalent

Point it at another bundle with ``--bundle`` or ``FLOW_BUNDLE=``.
"""

import argparse
import os
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ml.artifact import cell_key, load

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DEMO_DAY = "ml/data/varied/run_cars1000_seed5.npz"

DEFAULT_BUNDLE = "ml/models/varied.joblib"


class PredictRequest(BaseModel):
    """Today-so-far observations plus the cell to forecast."""

    target: Literal["speed", "counts"]
    horizon_s: float
    # [bins_since_midnight, n_edges]; speed is null where nothing drove.
    speed: list[list[float | None]]
    counts: list[list[float]]


def create_app(bundle_path: str | None = None,
               demo_day: str | None = None) -> FastAPI:
    """Build the service around one bundle (factory — tests pass a path;
    ``uvicorn ml.serve:create_app --factory`` reads ``FLOW_BUNDLE``).

    ``demo_day`` (or env ``FLOW_DEMO_DAY``) points at one recorded day's
    ``.npz`` — a *held-out test* day the model never trained on — which
    ``GET /demo/day`` streams to the frontend so the map can replay it and
    query forecasts against it. Optional: without it the map still draws,
    only the playback controls are disabled.
    """
    path = bundle_path or os.environ.get("FLOW_BUNDLE", DEFAULT_BUNDLE)
    demo_path = demo_day or os.environ.get("FLOW_DEMO_DAY", DEFAULT_DEMO_DAY)
    demo_cache: dict = {}
    bundle = load(path)
    models, statics, meta = (bundle["models"], bundle["statics"],
                             bundle["meta"])
    n_edges = meta["n_edges"]
    bin_s = meta["bin_s"]

    app = FastAPI(
        title="Traffic flow forecast",
        description="Per-street speed/occupancy forecasts from the "
                    "simulated city's fitted flow model.",
    )

    @app.get("/", include_in_schema=False)
    def index():
        """The map frontend: one self-contained HTML page (no build step),
        read per request so an edit shows on refresh during development."""
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404,
                                detail="frontend page missing from package")
        return HTMLResponse(page.read_text())

    @app.get("/health")
    def health():
        return {"status": "ok", "bundle": str(path), **meta}

    @app.get("/network")
    def network():
        """The street map: every edge as a drawable segment plus its static
        facts — everything a frontend needs to render the city, straight
        from the bundle (the network is fixed by the dataset contract)."""
        if "node_x" not in statics:
            raise HTTPException(
                status_code=404,
                detail="bundle has no geometry; run bin/python -m "
                       "ml.artifact --attach-geometry <bundle>")
        nx, ny = statics["node_x"], statics["node_y"]
        u, v = statics["edge_u"], statics["edge_v"]
        edges = [{
            "id": e,
            "x1": round(float(nx[u[e]]), 2), "y1": round(float(ny[u[e]]), 2),
            "x2": round(float(nx[v[e]]), 2), "y2": round(float(ny[v[e]]), 2),
            "speed_limit": round(float(statics["edge_speed_limit"][e]), 2),
            "length": round(float(statics["edge_length"][e]), 2),
            "lanes": int(statics["edge_lanes"][e]),
            "level": int(statics["edge_level"][e]),
            "zone": int(statics["edge_zone"][e]),
        } for e in range(n_edges)]
        return {
            "n_edges": n_edges,
            "bin_s": bin_s,
            "bounds": {"x_min": round(float(nx.min()), 2),
                       "x_max": round(float(nx.max()), 2),
                       "y_min": round(float(ny.min()), 2),
                       "y_max": round(float(ny.max()), 2)},
            "zone_codes": meta.get("zone_codes", {}),
            "edges": edges,
        }

    @app.get("/demo/day")
    def demo_day_data():
        """One recorded held-out day, for the frontend's replay: per-bin
        per-street observations exactly as ``POST /predict`` expects them
        back (``null`` speed = nobody drove). Loaded and JSON-shaped once,
        then cached."""
        if "payload" not in demo_cache:
            p = Path(demo_path)
            if not p.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"no demo day at {demo_path}; generate one with "
                           "bin/python -m ml.dataset (or set FLOW_DEMO_DAY)")
            with np.load(p) as z:
                speed, counts = z["speed"], z["counts"]
                bin_t = z["bin_t"]
            if speed.shape[1] != n_edges:
                raise HTTPException(status_code=500,
                                    detail="demo day edge count differs "
                                           "from the bundle's network")
            demo_cache["payload"] = {
                "n_bins": int(speed.shape[0]),
                "bin_s": bin_s,
                "day_length": meta["day_length"],
                "bin_t": [round(float(t), 1) for t in bin_t],
                "speed": [[None if np.isnan(x) else round(float(x), 2)
                           for x in row] for row in speed],
                "counts": [[round(float(x), 3) for x in row]
                           for row in counts],
            }
        return demo_cache["payload"]

    @app.post("/predict")
    def predict(req: PredictRequest):
        key = cell_key(req.target, req.horizon_s)
        if key not in models:
            raise HTTPException(
                status_code=404,
                detail=f"no model for {key}; available: "
                       f"{sorted(models)}")
        speed = np.array([[np.nan if x is None else x for x in row]
                          for row in req.speed], np.float32)
        counts = np.array(req.counts, np.float32)
        if speed.shape != counts.shape or speed.ndim != 2 \
                or speed.shape[1] != n_edges:
            raise HTTPException(
                status_code=400,
                detail=f"speed and counts must both be "
                       f"[bins, {n_edges}]; got {speed.shape} and "
                       f"{counts.shape}")
        model = models[key]
        try:
            pred = model.predict_next({**statics, "speed": speed,
                                       "counts": counts})
        except ValueError as err:            # short/overlong history
            raise HTTPException(status_code=400, detail=str(err))
        b_now = speed.shape[0] - 1
        return {
            "target": req.target, "horizon_s": req.horizon_s,
            "t_observed": b_now * bin_s,
            "t_forecast": b_now * bin_s + req.horizon_s,
            "predictions": [round(float(x), 4) for x in pred],
        }

    return app


def main(argv=None):
    """CLI: serve one bundle with uvicorn."""
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    uvicorn.run(create_app(args.bundle), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
