"""Reproducible experiments built on the traffic simulator.

Each module exposes a ``run(...)`` returning a list of result rows (so it can be
smoke-tested and reused) plus a ``render(...)`` that saves a figure, and a
``__main__`` that runs the default sweep and writes the figure under
``experiments/figures/`` (git-ignored — regenerate any time).
"""
