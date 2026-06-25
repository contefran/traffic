"""Matplotlib rendering: a static network plot and a live animation.

All car positions come from :meth:`RoadNetwork.point_on_edge`; this module
never recomputes geometry itself.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .network import DEFAULT_SPEED_LIMIT


GREEN = "#2ca02c"
RED = "#d62728"
ARTERIAL = "#1f77b4"


class Visuals:
    def _draw_edges(self, ax, net, arrows: bool = False) -> None:
        # Edges present in both directions are two-way; the rest are one-way and
        # get a direction arrow so the asymmetry is visible.
        pairs = {(e.u, e.v) for e in net.edges}
        for e in net.edges:
            n1, n2 = net.nodes[e.u], net.nodes[e.v]
            arterial = e.speed_limit > DEFAULT_SPEED_LIMIT + 1e-6
            ax.plot([n1.x, n2.x], [n1.y, n2.y],
                    color=ARTERIAL if arterial else "black",
                    linewidth=2.6 if arterial else 1.0,
                    zorder=1)
            one_way = (e.v, e.u) not in pairs
            if arrows or one_way:
                ax.annotate(
                    "", xytext=(n1.x, n1.y),
                    xy=(n1.x + 0.6 * (n2.x - n1.x), n1.y + 0.6 * (n2.y - n1.y)),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    alpha=0.35 if arrows else 0.7, lw=1.0),
                    zorder=1,
                )

    def _draw_signals(self, ax, net, signals, t: float) -> None:
        """Two indicators per signalized node: a horizontal one (E-W movement)
        and a vertical one (N-S), each green or red for the current phase."""
        from .signals import Orientation

        unit = net.edges[0].length if net.edges else 1.0
        off = 0.16 * unit
        hx, hy, hc, vx, vy, vc = [], [], [], [], [], []
        for node in net.nodes:
            if not signals.is_signalized(node.id):
                continue
            green = signals.controller.green_orientation(node.id, t)
            hx.append(node.x + off); hy.append(node.y)
            hc.append(GREEN if green is Orientation.HORIZONTAL else RED)
            vx.append(node.x); vy.append(node.y + off)
            vc.append(GREEN if green is Orientation.VERTICAL else RED)
        ax.scatter(hx, hy, c=hc, s=28, marker="s", zorder=4)
        ax.scatter(vx, vy, c=vc, s=28, marker="s", zorder=4)

    def render_state(self, net, cars=None, t: float = 0.0, signals=None,
                     path: str = "frame.png", title: str = None):
        """Render a single headless frame to ``path`` (PNG). Returns the path."""
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_edges(ax, net)
        ax.scatter([n.x for n in net.nodes], [n.y for n in net.nodes],
                   color="black", s=6, zorder=2)
        if signals is not None:
            self._draw_signals(ax, net, signals, t)
        if cars:
            xs, ys = zip(*(net.point_on_edge(c.edge_id, c.s) for c in cars))
            ax.scatter(xs, ys, s=55, color="tab:blue", zorder=5)

        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 12, max_x + 12)
        ax.set_ylim(min_y - 12, max_y + 12)
        ax.set_title(title or f"t = {t:.1f}s")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path

    def render_metrics(self, metrics, signals=None, net=None,
                       path: str = "metrics.png", title: str = None):
        """Plot mean speed and queue length over time, shaded by signal phase.

        Returns the PNG path. ``signals``/``net`` are optional; if given, the
        background is shaded by which orientation has green (fixed-time, all
        nodes in phase), so the effect of red/green on flow is visible.
        """
        from .signals import Orientation

        t = metrics.times
        fig, (ax_v, ax_q) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self._shade_phases(ax_v, t, signals, net)
        self._shade_phases(ax_q, t, signals, net)

        ax_v.plot(t, metrics.mean_speeds, color="tab:blue")
        ax_v.set_ylabel("mean speed [m/s]")
        ax_q.plot(t, metrics.queue_lengths, color="tab:red")
        ax_q.set_ylabel("queue length [cars]")
        ax_q.set_xlabel("time [s]")
        ax_v.set_title(title or "Traffic-flow metrics over time")
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path

    def _shade_phases(self, ax, times, signals, net) -> None:
        from .signals import Orientation

        if signals is None or net is None or not times:
            return
        node = next((n.id for n in net.nodes if signals.is_signalized(n.id)), None)
        if node is None:
            return
        # Shade contiguous runs where horizontal has green.
        start = times[0]
        prev = signals.controller.green_orientation(node, times[0]) is Orientation.HORIZONTAL
        for tt in times[1:] + [times[-1]]:
            cur = signals.controller.green_orientation(node, tt) is Orientation.HORIZONTAL
            if cur != prev:
                if prev:
                    ax.axvspan(start, tt, color=GREEN, alpha=0.08)
                else:
                    ax.axvspan(start, tt, color=RED, alpha=0.06)
                start, prev = tt, cur
        ax.axvspan(start, times[-1], color=(GREEN if prev else RED),
                   alpha=0.08 if prev else 0.06)

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

    def _signal_layout(self, net, signals):
        """Fixed marker positions for each signalized node's two indicators.

        Returns (positions, node_ids, orientations) where positions[k] belongs
        to node_ids[k] and shows orientations[k]'s state; colours are recomputed
        per frame from the controller.
        """
        unit = net.edges[0].length if net.edges else 1.0
        off = 0.16 * unit
        from .signals import Orientation

        positions, node_ids, orients = [], [], []
        for node in net.nodes:
            if not signals.is_signalized(node.id):
                continue
            positions.append((node.x + off, node.y))  # E-W indicator
            node_ids.append(node.id)
            orients.append(Orientation.HORIZONTAL)
            positions.append((node.x, node.y + off))  # N-S indicator
            node_ids.append(node.id)
            orients.append(Orientation.VERTICAL)
        return np.array(positions), node_ids, orients

    def _signal_colors(self, signals, t, node_ids, orients):
        return [GREEN if signals.controller.green_orientation(n, t) is o else RED
                for n, o in zip(node_ids, orients)]

    def _build_animation(self, net, sim, dt, steps):
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_edges(ax, net)
        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(min_y - 10, max_y + 10)

        signals = sim.signals
        sig_scat = None
        if signals is not None:
            pos, node_ids, orients = self._signal_layout(net, signals)
            sig_scat = ax.scatter(pos[:, 0], pos[:, 1], s=28, marker="s", zorder=4)

        scat = ax.scatter([], [], s=40, color="tab:blue", zorder=5)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            return (scat, sig_scat) if sig_scat else (scat,)

        def update(frame):
            sim.step(dt)
            points = [net.point_on_edge(c.edge_id, c.s) for c in sim.cars]
            scat.set_offsets(np.array(points) if points else np.empty((0, 2)))
            if sig_scat is not None:
                sig_scat.set_color(self._signal_colors(signals, sim.t, node_ids, orients))
                return (scat, sig_scat)
            return (scat,)

        anim = FuncAnimation(fig, update, frames=steps, init_func=init,
                             interval=dt * 1000, blit=True, repeat=False)
        return fig, anim

    def animate_sim(self, net, sim, dt: float = 0.1, steps: int = 400):
        """Open a live animation window (signal colours update each frame)."""
        _, anim = self._build_animation(net, sim, dt, steps)
        plt.show()
        return anim

    def save_animation(self, net, sim, path: str, dt: float = 0.1, steps: int = 400, fps: int = 20):
        """Render the simulation to a GIF at ``path`` (headless, no window)."""
        from matplotlib.animation import PillowWriter

        fig, anim = self._build_animation(net, sim, dt, steps)
        anim.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return path
