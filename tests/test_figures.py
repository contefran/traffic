"""Tests for the diagnostic figures (headless)."""

import os

import pytest

from traffic_sim import (
    build_grid_network,
    Car,
    ShortestPathRouter,
    RandomRouter,
    TrafficSim,
    MetricsCollector,
    Visuals,
)


def _run(record_edges, router, steps=200, n=10):
    net = build_grid_network(5, 5, block=120.0)
    cars = [Car(id=k, edge_id=k % len(net.edges), s=0.0, v=0.0) for k in range(n)]
    m = MetricsCollector(record_edges=record_edges)
    sim = TrafficSim(net, cars, router(net, seed=0), metrics=m)
    for _ in range(steps):
        sim.step(0.1)
    return net, m


def test_fundamental_diagram_writes_file(tmp_path):
    net, m = _run(record_edges=True, router=RandomRouter)
    out = str(tmp_path / "fd.png")
    assert Visuals().render_fundamental_diagram(m, net, path=out) == out
    assert os.path.getsize(out) > 0


def test_fundamental_diagram_requires_edge_recording(tmp_path):
    net, m = _run(record_edges=False, router=RandomRouter)
    with pytest.raises(ValueError):
        Visuals().render_fundamental_diagram(m, net, path=str(tmp_path / "fd.png"))


def test_delay_distribution_writes_file(tmp_path):
    net, m = _run(record_edges=False, router=ShortestPathRouter, steps=1500)
    out = str(tmp_path / "delay.png")
    assert Visuals().render_delay_distribution(m, path=out) == out
    assert os.path.getsize(out) > 0


def test_delay_distribution_requires_trips(tmp_path):
    net, m = _run(record_edges=False, router=RandomRouter)  # wanderer -> no trips
    with pytest.raises(ValueError):
        Visuals().render_delay_distribution(m, path=str(tmp_path / "delay.png"))
