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
    """Matplotlib rendering for the network, live sim, and metrics.

    Every method that shows cars reads their world position from
    :meth:`RoadNetwork.point_on_edge` and never recomputes geometry. The class
    offers three kinds of output: static frames (:meth:`render_state`,
    :meth:`plot_network`), a metrics time-series plot (:meth:`render_metrics`),
    and animations (:meth:`animate_sim` for a live window, :meth:`save_animation`
    for a headless GIF). It is imported lazily by the package so the simulation
    core has no hard matplotlib dependency.
    """

    def _draw_edges(self, ax, net, arrows: bool = False) -> None:
        """Draw all edges onto ``ax``.

        Arterials (higher speed limit) are drawn thicker and blue. Edges without
        a reverse partner are one-way and get a direction arrow so the asymmetry
        is visible; ``arrows=True`` forces (fainter) arrows on every edge.
        """
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

    def _signal_specs(self, net, signals):
        """Per signalized node, four movement indicators: E-W through/left to the
        right of the node, N-S through/left above it. Returns (positions, specs)
        where each spec is (node_id, orientation, turn) for colouring."""
        from .signals import Orientation, TurnType

        unit = net.edges[0].length if net.edges else 1.0
        off = 0.18 * unit
        positions, specs = [], []
        for node in net.nodes:
            if not signals.is_signalized(node.id):
                continue
            layout = [
                ((node.x + off, node.y - 0.4 * off), Orientation.HORIZONTAL, TurnType.STRAIGHT),
                ((node.x + off, node.y + 0.4 * off), Orientation.HORIZONTAL, TurnType.LEFT),
                ((node.x - 0.4 * off, node.y + off), Orientation.VERTICAL, TurnType.STRAIGHT),
                ((node.x + 0.4 * off, node.y + off), Orientation.VERTICAL, TurnType.LEFT),
            ]
            for pos, orient, turn in layout:
                positions.append(pos)
                specs.append((node.id, orient, turn))
        return positions, specs

    def _signal_colors(self, signals, t, specs):
        """Green/red colour for each indicator spec at time ``t``.

        ``specs`` is the ``(node_id, orientation, turn)`` list from
        :meth:`_signal_specs`; each entry is green iff ``signals`` currently
        allows that movement. Recomputed every animation frame.
        """
        return [GREEN if signals.allows(n, o, turn, t) else RED
                for (n, o, turn) in specs]

    def _draw_signals(self, ax, net, signals, t: float) -> None:
        """Scatter the per-node signal indicators onto ``ax`` for a static frame."""
        positions, specs = self._signal_specs(net, signals)
        if not positions:
            return
        pos = np.array(positions)
        ax.scatter(pos[:, 0], pos[:, 1], c=self._signal_colors(signals, t, specs),
                   s=16, marker="s", zorder=4)

    def _signal_legend(self, ax) -> None:
        """Add a key explaining the four signal squares drawn at each junction.

        Each signalized node shows *four* squares — one per movement, not a
        single light — which is easy to misread. This legend spells out which
        square is which: the east (right-hand) pair is the E-W approach (lower =
        through, upper = left); the north (top) pair is the N-S approach (left =
        through, right = left). Exactly one square is green at a time under the
        protected-phase controller (green = may go, red = stop). Anchored outside
        the plot so it never covers traffic.
        """
        from matplotlib.lines import Line2D

        def square(color, label):
            return Line2D([], [], marker="s", linestyle="None", markersize=8,
                          markerfacecolor=color, markeredgecolor="0.3", label=label)

        handles = [
            square("0.55", "top-left  =  N–S through"),
            square("0.55", "top-right  =  N–S left"),
            square("0.55", "right, lower  =  E–W through"),
            square("0.55", "right, upper  =  E–W left"),
            square(GREEN, "green  =  may go"),
            square(RED, "red  =  stop"),
        ]
        ax.legend(handles=handles,
                  title="Signal squares\n(4 per junction, 1 green at a time)",
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, title_fontsize=8, framealpha=0.95,
                  borderaxespad=0.0)

    def render_state(self, net, cars=None, t: float = 0.0, signals=None,
                     path: str = "frame.png", title: str = None):
        """Render a single headless frame to ``path`` (PNG). Returns the path.

        When ``signals`` is given, a legend keys the four per-junction signal
        squares (see :meth:`_signal_legend`); ``bbox_inches="tight"`` keeps it in
        the saved image.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_edges(ax, net)
        ax.scatter([n.x for n in net.nodes], [n.y for n in net.nodes],
                   color="black", s=6, zorder=2)
        if signals is not None:
            self._draw_signals(ax, net, signals, t)
            self._signal_legend(ax)
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
        """Shade the background of ``ax`` by signal phase over ``times``.

        Picks one representative signalized node and shades contiguous spans
        green where its east-west through movement is allowed and red otherwise,
        so the correlation between phase and flow is visible on the metrics plot.
        A no-op when there are no signals, no network, or no time samples.
        """
        from .signals import Orientation, TurnType

        if signals is None or net is None or not times:
            return
        node = next((n.id for n in net.nodes if signals.is_signalized(n.id)), None)
        if node is None:
            return
        # Shade contiguous runs where the E-W through movement has green.
        def h_through(tt):
            """Whether the representative node's east-west through is green at ``tt``."""
            return signals.allows(node, Orientation.HORIZONTAL, TurnType.STRAIGHT, tt)
        start = times[0]
        prev = h_through(times[0])
        for tt in times[1:] + [times[-1]]:
            cur = h_through(tt)
            if cur != prev:
                if prev:
                    ax.axvspan(start, tt, color=GREEN, alpha=0.08)
                else:
                    ax.axvspan(start, tt, color=RED, alpha=0.06)
                start, prev = tt, cur
        ax.axvspan(start, times[-1], color=(GREEN if prev else RED),
                   alpha=0.08 if prev else 0.06)

    def plot_network(self, net, cars=None) -> None:
        """Open an interactive window showing the network (and optional cars).

        A quick inspection helper: draws every edge with direction arrows, marks
        the nodes, and scatters any ``cars`` at their current positions, then
        blocks on ``plt.show()``. For headless output use :meth:`render_state`.
        """
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

    def _build_animation(self, net, sim, dt, steps):
        """Construct the matplotlib ``FuncAnimation`` that drives the sim.

        Sets up the static backdrop (edges, signal markers), then returns
        ``(fig, anim)`` where each frame calls ``sim.step(dt)``, moves the car
        scatter to the new positions, and recolours the signal markers. Shared by
        :meth:`animate_sim` (live window) and :meth:`save_animation` (GIF), which
        differ only in how they consume the returned animation. Runs for
        ``steps`` frames.
        """
        fig, ax = plt.subplots(figsize=(8.6, 6))
        self._draw_edges(ax, net)
        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(min_y - 10, max_y + 10)

        signals = sim.signals
        sig_scat = None
        if signals is not None:
            positions, specs = self._signal_specs(net, signals)
            if positions:
                pos = np.array(positions)
                sig_scat = ax.scatter(pos[:, 0], pos[:, 1], s=16, marker="s", zorder=4)
                self._signal_legend(ax)
                # Reserve room on the right so the legend sits beside the map.
                fig.subplots_adjust(left=0.06, right=0.66)

        scat = ax.scatter([], [], s=40, color="tab:blue", zorder=5)

        def init():
            """Blit initialiser: start with an empty car scatter."""
            scat.set_offsets(np.empty((0, 2)))
            return (scat, sig_scat) if sig_scat else (scat,)

        def update(frame):
            """Advance the sim one step and redraw cars (and signal colours)."""
            sim.step(dt)
            points = [net.point_on_edge(c.edge_id, c.s) for c in sim.cars]
            scat.set_offsets(np.array(points) if points else np.empty((0, 2)))
            if sig_scat is not None:
                sig_scat.set_color(self._signal_colors(signals, sim.t, specs))
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
