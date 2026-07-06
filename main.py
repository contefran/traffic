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
    PermissiveLeftModel,
    assign_zones,
    apply_zone_speeds,
    add_grade_separated,
    add_roundabouts,
    DemandModel,
    ParkingModel,
    ActivitySchedule,
    assign_venues,
    MetricsCollector,
    kmh_to_ms,
    ms_to_kmh,
    Visuals,
)
from traffic_sim.network import DEFAULT_SPEED_LIMIT


def spawn_cars(net, n_cars: int, seed: int = 0):
    """Place ``n_cars`` at random positions on random edges (deterministic).

    Cars start stationary, so two placed within a car-length of each other on the
    same edge and lane would be an instant crash on step 1 — a pure placement
    artefact, not traffic. We reject any position closer than ``min_gap`` to an
    already-placed car in the same (edge, lane), retrying a few times before
    moving to a fresh edge, so the initial state is always collision-free.
    """
    rng = random.Random(seed)
    min_gap = Car.length + Car.s0 + 2.0        # rear + standstill gap + margin
    occupied: dict = {}                         # (edge_id, lane) -> [s, ...]
    cars = []
    for i in range(n_cars):
        for _ in range(20):                     # a few tries for a clear slot
            edge = rng.choice(net.edges)
            lane = rng.randrange(edge.lanes)
            s = rng.uniform(0.0, edge.length)
            placed = occupied.setdefault((edge.id, lane), [])
            if all(abs(s - o) >= min_gap for o in placed):
                placed.append(s)
                cars.append(Car(id=i, edge_id=edge.id, s=s, v=0.0, lane=lane))
                break
        else:                                   # no clear slot found; place anyway
            cars.append(Car(id=i, edge_id=edge.id, s=s, v=0.0, lane=lane))
    return cars


def assign_homes(cars, zones, seed: int = 0):
    """Give every car a ``home`` (a residential street it returns to).

    Residential streets are dealt out so that **every one gets at least one home**
    before any is reused: the shuffled list of residential edges is handed to the
    first cars one-each, and any remaining cars get a random residential street.
    A no-op when there are no residential streets. Deterministic under ``seed``.
    """
    from traffic_sim import LandUse

    residential = [eid for eid, use in zones.items() if use is LandUse.RESIDENTIAL]
    if not residential:
        return
    rng = random.Random(seed)
    order = residential[:]
    rng.shuffle(order)
    for i, car in enumerate(cars):
        car.home = order[i] if i < len(order) else rng.choice(residential)


def pick_roundabout_nodes(net, count: int, seed: int = 0):
    """Choose ``count`` interior local crossings to turn into roundabouts.

    Prefers busy **ground** intersections (≥3 in- and out-edges) that are away
    from the border and touch no arterial (roundabouts sit on local street
    crossings here, not on fast through-roads), and spaces them out so no two are
    grid-adjacent. Deterministic under ``seed``.
    """
    width = max(n.i for n in net.nodes) + 1
    height = max(n.j for n in net.nodes) + 1

    def local_crossing(n):
        """A busy interior ground junction of local streets (no arterials)."""
        if n.level != 0 or n.internal:
            return False
        if not (0 < n.i < width - 1 and 0 < n.j < height - 1):
            return False
        if len(n.in_edges) < 3 or len(n.out_edges) < 3:
            return False
        incident = n.in_edges + n.out_edges
        return all(net.edges[e].speed_limit <= DEFAULT_SPEED_LIMIT + 1e-6
                   for e in incident)

    candidates = [n for n in net.nodes if local_crossing(n)]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen, used = [], []
    for n in candidates:
        if len(chosen) >= count:
            break
        if any(abs(n.i - m.i) + abs(n.j - m.j) <= 2 for m in used):
            continue                    # keep roundabouts spread apart
        chosen.append(n.id)
        used.append(n)
    return chosen


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (see module docstring for the big picture)."""
    p = argparse.ArgumentParser(
        description="Run the traffic simulator on a heterogeneous city grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    net = p.add_argument_group("network")
    net.add_argument("--width", type=int, default=20, help="grid columns")
    net.add_argument("--height", type=int, default=20, help="grid rows")
    net.add_argument("--block", type=float, default=150.0, help="block spacing [m]")
    net.add_argument("--seed", type=int, default=1, help="network RNG seed")
    net.add_argument("--jitter", type=float, default=0.22,
                     help="node position jitter, as a fraction of block")
    net.add_argument("--one-way-prob", type=float, default=0.15,
                     help="probability a connection is one-way")
    net.add_argument("--drop-prob", type=float, default=0.28,
                     help="probability a (non-arterial) connection is missing "
                          "(higher = more T-junctions, fewer full 4-way crossings)")
    net.add_argument("--arterial-every", type=int, default=3,
                     help="every Nth row/column is an arterial (0 = none)")
    net.add_argument("--arterial-speed", type=float, default=70.0,
                     help="arterial speed limit [km/h]")
    net.add_argument("--arterial-lanes", type=int, default=2, help="lanes per arterial")
    net.add_argument("--ring", action=argparse.BooleanOptionalAction, default=True,
                     help="make the perimeter a fast multi-lane ring road")
    net.add_argument("--ring-speed", type=float, default=100.0,
                     help="ring speed limit [km/h]")
    net.add_argument("--ring-lanes", type=int, default=3, help="lanes on the ring")
    net.add_argument("--ring-access-spacing", type=float, default=1000.0,
                     help="metres between ring on/off ramps (>=1 per side)")
    net.add_argument("--grade", action=argparse.BooleanOptionalAction, default=True,
                     help="elevate the ring + a cross-city expressway (grade-separated)")
    net.add_argument("--roundabouts", type=int, default=12,
                     help="number of local crossings to build as geometric roundabouts")

    demand = p.add_argument_group("demand & land use")
    demand.add_argument("--demand", action=argparse.BooleanOptionalAction, default=True,
                        help="time-of-day zone-based destinations (else uniform random)")
    demand.add_argument("--parking", action=argparse.BooleanOptionalAction, default=True,
                        help="cars park at destinations and reappear after a dwell")
    demand.add_argument("--day-length", type=float, default=1200.0,
                        help="simulated seconds in one day (four periods); long "
                             "enough that a physical commute is a realistic clock fraction")
    demand.add_argument("--residential-speed", action=argparse.BooleanOptionalAction,
                        default=True, help="slow local streets in residential zones to 30 km/h")
    demand.add_argument("--edge-points", action=argparse.BooleanOptionalAction,
                        default=True, help="destinations are mid-block points, not junctions")
    demand.add_argument("--schedule", action=argparse.BooleanOptionalAction, default=True,
                        help="per-car activity plans (home->work->maybe out->home), "
                             "computed for the whole day and executed from midnight")
    demand.add_argument("--work-scale", type=float, default=400.0,
                        help="Gaussian home->work distance scale [m]: people take "
                             "jobs near home (smaller = shorter commutes). A key knob "
                             "for analysis.")

    traffic = p.add_argument_group("traffic")
    traffic.add_argument("--cars", type=int, default=1000, help="number of cars")
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
    run.add_argument("--steps", type=int, default=None,
                     help="number of steps (default: one full day = day-length / dt, "
                          "so the run goes midnight -> midnight)")
    run.add_argument("--save-gif", metavar="PATH", default=None,
                     help="render headless to this GIF instead of a live window")
    run.add_argument("--fps", type=int, default=30,
                     help="target frame rate for the live window and the GIF")
    run.add_argument("--steps-per-frame", type=int, default=1,
                     help="advance N sim steps per rendered frame (Nx faster "
                          "playback; higher = choppier motion)")
    run.add_argument("--no-show-signals", dest="show_signals", action="store_false",
                     help="hide the traffic-light markers (declutters big maps; "
                          "signals still operate)")

    return p


def build_simulation(args):
    """Assemble ``(net, sim, zones)`` from parsed ``args``.

    Speeds in ``args`` are km/h and are converted to SI here (the only place the
    boundary conversion happens). A :class:`MetricsCollector` is always attached
    so the run reports its flow statistics. The land-use ``zones`` map is returned
    alongside so the visualiser can colour the demo by zone.
    """
    net = build_city_grid(
        width=args.width, height=args.height, block=args.block,
        seed=args.seed, jitter=args.jitter,
        one_way_prob=args.one_way_prob, drop_prob=args.drop_prob,
        arterial_every=args.arterial_every,
        arterial_speed=kmh_to_ms(args.arterial_speed),
        arterial_lanes=args.arterial_lanes,
        ring=args.ring and not args.grade,   # 2-D ring only when not elevated
        ring_speed=kmh_to_ms(args.ring_speed),
        ring_lanes=args.ring_lanes, ring_access_spacing=args.ring_access_spacing,
    )
    if args.grade:
        add_grade_separated(net, block=args.block, speed=kmh_to_ms(args.ring_speed),
                            lanes=args.ring_lanes, access_spacing=args.ring_access_spacing)

    # Geometric roundabouts on some local crossings (before zoning, so their ring
    # edges stay unzoned). Ring nodes are unsignalized and circulating traffic has
    # right-of-way (both wired below).
    ring_nodes, circulating = set(), set()
    if args.roundabouts > 0:
        centres = pick_roundabout_nodes(net, args.roundabouts, seed=args.seed)
        ring_nodes, circulating = add_roundabouts(net, centres)

    # Land use drives demand, dwell times, and (optionally) residential speeds.
    zones = assign_zones(net, seed=args.seed)
    if args.residential_speed:
        apply_zone_speeds(net, zones)

    demand = (DemandModel(net, zones, seed=args.router_seed, day_length=args.day_length)
              if args.demand else None)
    cars = spawn_cars(net, args.cars, seed=args.car_seed)
    assign_homes(cars, zones, seed=args.car_seed)   # each car returns to its own house
    # Activity-based per-car schedules: each car gets a full periodic day (home ->
    # work -> maybe lunch/gym/dining/pub -> home), computed once and executed from
    # midnight, so the rush — and the nightlife — emerge from the agents. Venues
    # (restaurants/gyms/pubs) are a retail-leaning overlay across all zones.
    schedule = None
    if args.schedule:
        venues = assign_venues(net, zones, seed=args.seed)
        schedule = ActivitySchedule(venues, day_length=args.day_length,
                                    seed=args.car_seed, work_scale=args.work_scale)
        schedule.assign(cars, net, zones)
    router = (ShortestPathRouter(net, seed=args.router_seed, demand=demand,
                                 edge_points=args.edge_points)
              if args.router == "shortest"
              else RandomRouter(net, seed=args.router_seed))
    controller = (ProtectedPhaseController(green_time=args.green_time, yellow=args.yellow)
                  if args.controller == "protected"
                  else FixedTimeController(green_time=args.green_time, yellow=args.yellow))
    # A ring/highway flows freely: unsignalize it so through traffic never stops,
    # and lower-priority traffic yields to merge (the priority model gives the
    # faster road right of way). Grade separation unsignalizes the whole elevated
    # level; the 2-D ring unsignalizes the perimeter.
    unsig = set()
    if args.grade:
        unsig |= {n.id for n in net.nodes if n.level == 1}
    elif args.ring:
        unsig |= {n.id for n in net.nodes
                  if n.i in (0, args.width - 1) or n.j in (0, args.height - 1)}
    unsig |= ring_nodes          # roundabout ring nodes are never signalized
    signals = SignalSystem(net, controller, unsignalized_nodes=unsig or None)
    # Priority gives circulating roundabout traffic right-of-way over entries.
    priority = PriorityModel(net, circulating=circulating) if args.priority else None
    left_turn = PermissiveLeftModel(net)   # inert under protected phasing
    parking = ParkingModel(seed=args.car_seed, zones=zones) if args.parking else None
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, router, signals=signals, priority=priority,
                     left_turn=left_turn, parking=parking, metrics=metrics,
                     schedule=schedule)
    return net, sim, zones


def main(argv=None) -> None:
    """Parse arguments, build the simulation, run it, and report metrics."""
    args = build_parser().parse_args(argv)
    # Default the run to exactly one day so the clock goes midnight -> midnight.
    if args.steps is None:
        args.steps = max(1, round(args.day_length / args.dt))
    net, sim, zones = build_simulation(args)
    print(f"nodes={len(net.nodes)} edges={len(net.edges)} cars={args.cars} "
          f"steps={args.steps} (day={args.day_length:.0f}s)")

    visuals = Visuals()
    if args.save_gif:
        visuals.save_animation(net, sim, args.save_gif,
                               dt=args.dt, steps=args.steps, fps=args.fps, zones=zones,
                               show_signals=args.show_signals, day_length=args.day_length,
                               steps_per_frame=args.steps_per_frame)
        print(f"saved animation to {args.save_gif}")
    else:
        visuals.animate_sim(net, sim, dt=args.dt, steps=args.steps, zones=zones,
                            show_signals=args.show_signals, day_length=args.day_length,
                            fps=args.fps, steps_per_frame=args.steps_per_frame)

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
