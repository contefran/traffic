"""Entry point: build a heterogeneous city grid, fill it with traffic, animate.

Run ``python main.py --help`` for the full list of options. Everything about a
run — grid size, heterogeneity, traffic density, the routing / signal / priority
policies, and whether to open a live window or save a GIF — is a command-line
flag, so experimenting no longer means editing source.

Speeds are given in **km/h** on the command line and converted to the
simulator's internal SI units (m/s) at the boundary; the core stays SI.
"""

import argparse
import random

from traffic_sim import (
    build_city_grid,
    Car,
    ShortestPathRouter,
    RandomRouter,
    TrafficSim,
    ProtectedPhaseController,
    FixedTimeController,
    SignalSystem,
    PriorityModel,
    MetricsCollector,
    kmh_to_ms,
    ms_to_kmh,
    Visuals,
)


def spawn_cars(net, n_cars: int, seed: int = 0):
    """Place ``n_cars`` at random positions on random edges (deterministic)."""
    rng = random.Random(seed)
    cars = []
    for i in range(n_cars):
        edge = rng.choice(net.edges)
        cars.append(Car(id=i, edge_id=edge.id, s=rng.uniform(0.0, edge.length), v=0.0))
    return cars


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (see module docstring for the big picture)."""
    p = argparse.ArgumentParser(
        description="Run the traffic simulator on a heterogeneous city grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    net = p.add_argument_group("network")
    net.add_argument("--width", type=int, default=8, help="grid columns")
    net.add_argument("--height", type=int, default=8, help="grid rows")
    net.add_argument("--block", type=float, default=150.0, help="block spacing [m]")
    net.add_argument("--seed", type=int, default=1, help="network RNG seed")
    net.add_argument("--jitter", type=float, default=0.22,
                     help="node position jitter, as a fraction of block")
    net.add_argument("--one-way-prob", type=float, default=0.15,
                     help="probability a connection is one-way")
    net.add_argument("--drop-prob", type=float, default=0.12,
                     help="probability a (non-arterial) connection is missing")
    net.add_argument("--arterial-every", type=int, default=3,
                     help="every Nth row/column is an arterial (0 = none)")
    net.add_argument("--arterial-speed", type=float, default=70.0,
                     help="arterial speed limit [km/h]")

    traffic = p.add_argument_group("traffic")
    traffic.add_argument("--cars", type=int, default=60, help="number of cars")
    traffic.add_argument("--car-seed", type=int, default=1, help="car placement seed")
    traffic.add_argument("--router", choices=("shortest", "random"), default="shortest",
                         help="routing policy")
    traffic.add_argument("--router-seed", type=int, default=42, help="router RNG seed")

    control = p.add_argument_group("control")
    control.add_argument("--controller", choices=("protected", "fixed"),
                         default="protected", help="signal controller")
    control.add_argument("--green-time", type=float, default=4.5,
                         help="green duration per phase [s]")
    control.add_argument("--yellow", type=float, default=1.5,
                         help="yellow/clearance interval per phase [s]")
    control.add_argument("--priority", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="right-of-way at unsignalized nodes")

    run = p.add_argument_group("run / output")
    run.add_argument("--dt", type=float, default=0.1, help="time step [s]")
    run.add_argument("--steps", type=int, default=800, help="number of steps")
    run.add_argument("--save-gif", metavar="PATH", default=None,
                     help="render headless to this GIF instead of a live window")
    run.add_argument("--fps", type=int, default=20, help="GIF frame rate")

    return p


def build_simulation(args):
    """Assemble ``(net, sim)`` from parsed ``args``.

    Speeds in ``args`` are km/h and are converted to SI here (the only place the
    boundary conversion happens). A :class:`MetricsCollector` is always attached
    so the run reports its flow statistics.
    """
    net = build_city_grid(
        width=args.width, height=args.height, block=args.block,
        seed=args.seed, jitter=args.jitter,
        one_way_prob=args.one_way_prob, drop_prob=args.drop_prob,
        arterial_every=args.arterial_every,
        arterial_speed=kmh_to_ms(args.arterial_speed),
    )

    cars = spawn_cars(net, args.cars, seed=args.car_seed)
    router = (ShortestPathRouter(net, seed=args.router_seed) if args.router == "shortest"
              else RandomRouter(net, seed=args.router_seed))
    controller = (ProtectedPhaseController(green_time=args.green_time, yellow=args.yellow)
                  if args.controller == "protected"
                  else FixedTimeController(green_time=args.green_time, yellow=args.yellow))
    signals = SignalSystem(net, controller)
    priority = PriorityModel(net) if args.priority else None
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, router, signals=signals, priority=priority,
                     metrics=metrics)
    return net, sim


def main(argv=None) -> None:
    """Parse arguments, build the simulation, run it, and report metrics."""
    args = build_parser().parse_args(argv)
    net, sim = build_simulation(args)
    print(f"nodes={len(net.nodes)} edges={len(net.edges)} cars={args.cars}")

    visuals = Visuals()
    if args.save_gif:
        visuals.save_animation(net, sim, args.save_gif,
                               dt=args.dt, steps=args.steps, fps=args.fps)
        print(f"saved animation to {args.save_gif}")
    else:
        visuals.animate_sim(net, sim, dt=args.dt, steps=args.steps)

    # Metrics were recorded every step; report them (speed shown in km/h too).
    s = sim.metrics.summary()
    if s.get("steps"):
        print(f"metrics: avg_speed={s['avg_speed']:.2f} m/s "
              f"({ms_to_kmh(s['avg_speed']):.1f} km/h), "
              f"avg_queue={s['avg_queue']:.1f}, max_queue={s['max_queue']}, "
              f"throughput={s['throughput_per_s']:.2f}/s, "
              f"crashes={s['crashes']}")
        if "mean_delay_s" in s:
            print(f"         trips={s['trips_completed']}, "
                  f"mean_delay={s['mean_delay_s']:.1f} s, "
                  f"stops/trip={s['mean_stops_per_trip']:.2f}, "
                  f"fuel={s['fuel_proxy']:.0f}")
    else:
        print("metrics:", s)


if __name__ == "__main__":
    main()
