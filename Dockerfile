# The flow-forecast service in a box: FastAPI (ml/serve.py) around the
# fitted model bundle, self-contained — `docker run -p 8000:8000 <image>`
# answers /health and /predict with no repo, venv, or dataset present.
#
# Build (the bundle must exist first — it is gitignored, built by
# `bin/python -m ml.artifact`):
#
#     docker build -t traffic-flow .
#     docker run --rm -p 8000:8000 traffic-flow
#     curl localhost:8000/health
#
# The bundle is baked into the image (it is small, ~2 MB, and an image
# that ships its exact model version is the point of the exercise); to
# serve a different bundle without rebuilding, mount it and point
# FLOW_BUNDLE at it:
#
#     docker run --rm -p 8000:8000 \
#         -v $PWD/ml/models:/app/ml/models:ro \
#         -e FLOW_BUNDLE=ml/models/other.joblib traffic-flow

FROM python:3.13-slim

# Dependencies first, code second: the pip layer (the slow, large one) is
# rebuilt only when requirements-serve.txt changes, not on every code edit.
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The code the bundle unpickles against (ml.sklearn_models -> ml.linear)
# plus the sim package it imports units from (pure stdlib, no matplotlib),
# and the bundle itself.
COPY traffic_sim/ traffic_sim/
COPY ml/ ml/
COPY ml/models/varied.joblib ml/models/varied.joblib

# Run as an unprivileged user; the app only ever reads from the image.
RUN useradd --create-home app
USER app

ENV FLOW_BUNDLE=ml/models/varied.joblib
EXPOSE 8000

CMD ["uvicorn", "ml.serve:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
