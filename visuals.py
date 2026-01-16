import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Visuals:

    def __init__(self) -> None:
        pass

    def plot_network(self, net, cars=None) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))

        # draw edges
        for e in net.edges:
            n1 = net.nodes[e.u]
            n2 = net.nodes[e.v]
            ax.plot([n1.x, n2.x], [n1.y, n2.y], color="black", linewidth=1)
            ax.arrow(
                n1.x, n1.y,
                (n2.x - n1.x)/1.5, (n2.y - n1.y)/1.5,
                length_includes_head=True,
                head_width=3,
                alpha=0.3
            )

        # draw nodes
        xs = [n.x for n in net.nodes]
        ys = [n.y for n in net.nodes]
        ax.scatter(xs, ys, color="red", s=10, zorder=3)

        if cars:
            xs, ys = zip(*(c.car_xy(net, c) for c in cars))
            ax.scatter(xs, ys, s=50)


        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Road network")
        plt.show()


    def animate_sim(self, net, sim, dt=0.2, steps=200):
        fig, ax = plt.subplots(figsize=(6, 6))

        # draw static road network
        for e in net.edges:
            n1 = net.nodes[e.u]
            n2 = net.nodes[e.v]
            ax.plot([n1.x, n2.x], [n1.y, n2.y], color="black", linewidth=1)

        ax.set_aspect("equal")
        ax.set_xlim(min(n.x for n in net.nodes) - 10,
                    max(n.x for n in net.nodes) + 10)
        ax.set_ylim(min(n.y for n in net.nodes) - 10,
                    max(n.y for n in net.nodes) + 10)

        # car artists (blue dots)
        scat = ax.scatter([], [], s=40, color="blue", zorder=3)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            return (scat,)

        def update(frame):
            sim.step(dt)

            xs, ys = [], []
            for car in sim.cars:
                e = net.edges[car.edge_id]
                n1 = net.nodes[e.u]
                n2 = net.nodes[e.v]
                t = car.s / e.length if e.length > 0 else 0.0
                xs.append(n1.x + t * (n2.x - n1.x))
                ys.append(n1.y + t * (n2.y - n1.y))

            scat.set_offsets(np.column_stack([xs, ys]))
            return (scat,)

        anim = FuncAnimation(
            fig,
            update,
            frames=steps,
            init_func=init,
            interval=dt * 1000,
            blit=True,
            repeat=False
        )

        plt.show()
