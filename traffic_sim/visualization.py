"""Matplotlib rendering: a static network plot and a live animation.

All car positions come from :meth:`RoadNetwork.point_on_edge`; this module
never recomputes geometry itself.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Visuals:
    def _draw_edges(self, ax, net, arrows: bool = False) -> None:
        for e in net.edges:
            n1, n2 = net.nodes[e.u], net.nodes[e.v]
            ax.plot([n1.x, n2.x], [n1.y, n2.y], color="black", linewidth=1)
            if arrows:
                ax.arrow(
                    n1.x, n1.y,
                    (n2.x - n1.x) / 1.5, (n2.y - n1.y) / 1.5,
                    length_includes_head=True, head_width=3, alpha=0.3,
                )

    def plot_network(self, net, cars=None) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_edges(ax, net, arrows=True)

        ax.scatter([n.x for n in net.nodes], [n.y for n in net.nodes],
                   color="red", s=10, zorder=3)

        if cars:
            xs, ys = zip(*(net.point_on_edge(c.edge_id, c.s) for c in cars))
            ax.scatter(xs, ys, s=50)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Road network")
        plt.show()

    def animate_sim(self, net, sim, dt: float = 0.1, steps: int = 400):
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_edges(ax, net)

        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(min_y - 10, max_y + 10)

        scat = ax.scatter([], [], s=40, color="blue", zorder=3)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            return (scat,)

        def update(frame):
            sim.step(dt)
            points = [net.point_on_edge(c.edge_id, c.s) for c in sim.cars]
            scat.set_offsets(np.array(points) if points else np.empty((0, 2)))
            return (scat,)

        anim = FuncAnimation(
            fig, update, frames=steps, init_func=init,
            interval=dt * 1000, blit=True, repeat=False,
        )
        plt.show()
        return anim
