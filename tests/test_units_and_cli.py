"""Tests for km/h<->m/s conversion and the main.py argparse CLI wiring."""

import math

from traffic_sim import kmh_to_ms, ms_to_kmh
from traffic_sim.network import DEFAULT_SPEED_LIMIT

import main


def test_kmh_ms_known_values():
    assert math.isclose(kmh_to_ms(90.0), 25.0)
    assert math.isclose(ms_to_kmh(25.0), 90.0)
    assert math.isclose(kmh_to_ms(50.0), 13.888888, rel_tol=1e-5)


def test_kmh_ms_round_trip():
    for v in (0.0, 13.9, 30.0, 50.0, 110.0):
        assert math.isclose(ms_to_kmh(kmh_to_ms(v)), v, rel_tol=1e-12)


def test_parser_defaults_match_the_documented_demo():
    args = main.build_parser().parse_args([])
    assert (args.width, args.height) == (20, 20)
    assert args.controller == "protected"
    assert args.priority is True
    assert args.router == "shortest"
    assert args.arterial_speed == 70.0  # km/h


def test_no_priority_flag_disables_priority():
    args = main.build_parser().parse_args(["--no-priority"])
    assert args.priority is False
    _, sim = main.build_simulation(args)
    assert sim.priority is None


def test_build_simulation_runs_and_records_metrics():
    # A small, fast configuration; step it and confirm metrics accumulate.
    args = main.build_parser().parse_args(
        ["--width", "5", "--height", "5", "--cars", "10", "--steps", "20"])
    net, sim = main.build_simulation(args)
    for _ in range(args.steps):
        sim.step(args.dt)
    assert len(sim.metrics.history) == args.steps
    assert sim.metrics.summary()["steps"] == args.steps


def test_arterial_speed_km_h_lands_as_m_s_on_arterials():
    # --arterial-speed is km/h; arterial edges should carry the converted m/s.
    args = main.build_parser().parse_args(["--arterial-speed", "90", "--arterial-every", "3"])
    net, _ = main.build_simulation(args)
    speeds = {round(e.speed_limit, 6) for e in net.edges}
    assert round(kmh_to_ms(90.0), 6) in speeds          # 25 m/s arterials
    assert round(DEFAULT_SPEED_LIMIT, 6) in speeds       # local streets unchanged


def test_random_router_selected_by_flag():
    from traffic_sim import RandomRouter
    args = main.build_parser().parse_args(["--router", "random"])
    _, sim = main.build_simulation(args)
    assert isinstance(sim.router, RandomRouter)
