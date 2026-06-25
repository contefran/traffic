"""Entry point: build a heterogeneous city grid, fill it with traffic, animate."""

import random

from traffic_sim import (
    build_city_grid,
    Car,
    RandomRouter,
    TrafficSim,
    FixedTimeController,
    SignalSystem,
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
    net = build_city_grid(
        width=8, height=8, block=60.0,
        seed=1, jitter=0.18, one_way_prob=0.15, arterial_every=3, arterial_speed=25.0,
    )
    print(f"nodes={len(net.nodes)} edges={len(net.edges)}")

    cars = spawn_cars(net, n_cars=60, seed=1)
    router = RandomRouter(net, seed=42)
    signals = SignalSystem(net, FixedTimeController(green_time=8.0))
    sim = TrafficSim(net, cars, router, signals=signals)

    Visuals().animate_sim(net, sim, dt=0.1, steps=800)


if __name__ == "__main__":
    main()
