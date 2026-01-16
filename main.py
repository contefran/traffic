from builder import Builder
from traffic import Car, TrafficSim
from visuals import Visuals

class Main:

    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    # build the road network
    builder = Builder(W=4, H=3, block=50.0)
    net = builder.build_grid_network()

    print(f"nodes={len(net.nodes)} edges={len(net.edges)}")
    #print("node(0,0) out_edges:", net.nodes[net.node_id[(0,0)]].out_edges)

    cars = [Car(id=0, edge_id=12, s=10.0, v=5.0), Car(1, 5, 20.0, 7.0)]

    # visualize the road network
    viz = Visuals()
    #viz.plot_network(net, cars=cars)

    sim = TrafficSim(net, cars)

    #dt = 0.02
    #for _ in range(200):
    #    sim.step(dt)

    viz.animate_sim(net, sim, dt=0.01, steps=200)

    #viz.plot_network(net, cars=cars)
