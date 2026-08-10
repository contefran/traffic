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
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.artifact import cell_key, load

DEFAULT_BUNDLE = "ml/models/varied.joblib"


class PredictRequest(BaseModel):
    """Today-so-far observations plus the cell to forecast."""

    target: Literal["speed", "counts"]
    horizon_s: float
    # [bins_since_midnight, n_edges]; speed is null where nothing drove.
    speed: list[list[float | None]]
    counts: list[list[float]]


def create_app(bundle_path: str | None = None) -> FastAPI:
    """Build the service around one bundle (factory — tests pass a path;
    ``uvicorn ml.serve:create_app --factory`` reads ``FLOW_BUNDLE``)."""
    path = bundle_path or os.environ.get("FLOW_BUNDLE", DEFAULT_BUNDLE)
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

    @app.get("/health")
    def health():
        return {"status": "ok", "bundle": str(path), **meta}

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
