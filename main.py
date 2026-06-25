"""Entry point: build a grid, place a few cars, and animate the simulation."""

from traffic_sim import build_grid_network, Car, RandomRouter, TrafficSim, Visuals


def main() -> None:
    net = build_grid_network(width=4, height=3, block=50.0)
    print(f"nodes={len(net.nodes)} edges={len(net.edges)}")

    cars = [
        Car(id=0, edge_id=12, s=10.0, v=5.0),
        Car(id=1, edge_id=5, s=20.0, v=7.0),
    ]

    router = RandomRouter(net, seed=42)
    sim = TrafficSim(net, cars, router)

    Visuals().animate_sim(net, sim, dt=0.1, steps=400)


if __name__ == "__main__":
    main()
