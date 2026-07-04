"""Entry point: build a heterogeneous city grid, fill it with traffic, animate."""

import random

from traffic_sim import (
    build_city_grid,
    Car,
    ShortestPathRouter,
    TrafficSim,
    ProtectedPhaseController,
    SignalSystem,
    PriorityModel,
    MetricsCollector,
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


def main() -> None:
    """Build a heterogeneous city grid, populate it, and open the animation.

    A worked example of wiring the pieces together: a jittered city grid with
    one-way streets, dropped links and arterials; 60 cars; a
    :class:`ShortestPathRouter` steering each to a destination; a
    :class:`ProtectedPhaseController` on the signals; and a
    :class:`PriorityModel` for right-of-way at any unsignalized node. Edit the
    values here to experiment with grid size, density, ``dt`` and step count.
    """
    net = build_city_grid(
        width=8, height=8, block=60.0,
        seed=1, jitter=0.22, one_way_prob=0.15, drop_prob=0.12,
        arterial_every=3, arterial_speed=25.0,
    )
    print(f"nodes={len(net.nodes)} edges={len(net.edges)}")

    cars = spawn_cars(net, n_cars=60, seed=1)
    router = ShortestPathRouter(net, seed=42)  # each car heads to a destination
    signals = SignalSystem(net, ProtectedPhaseController(green_time=5.0))
    # Right-of-way at any unsignalized node (arterial priority + gap acceptance).
    priority = PriorityModel(net)
    # Record flow diagnostics (speed, queues, throughput) every step.
    metrics = MetricsCollector()
    sim = TrafficSim(net, cars, router, signals=signals, priority=priority,
                     metrics=metrics)

    Visuals().animate_sim(net, sim, dt=0.1, steps=800)

    # The run is over (window closed or all steps done): report what it produced.
    print("metrics:", sim.metrics.summary())


if __name__ == "__main__":
    main()
