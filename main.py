"""Entry point: build a grid, place cars under traffic signals, and animate."""

from traffic_sim import (
    build_grid_network,
    Car,
    RandomRouter,
    TrafficSim,
    FixedTimeController,
    SignalSystem,
    Visuals,
)


def main() -> None:
    net = build_grid_network(width=4, height=3, block=50.0)
    print(f"nodes={len(net.nodes)} edges={len(net.edges)}")

    cars = [
        Car(id=0, edge_id=12, s=10.0, v=5.0),
        Car(id=1, edge_id=5, s=20.0, v=7.0),
        Car(id=2, edge_id=0, s=5.0, v=6.0),
        Car(id=3, edge_id=0, s=18.0, v=6.0),
    ]

    router = RandomRouter(net, seed=42)
    signals = SignalSystem(net, FixedTimeController(green_time=8.0))
    sim = TrafficSim(net, cars, router, signals=signals)

    Visuals().animate_sim(net, sim, dt=0.1, steps=600)


if __name__ == "__main__":
    main()
