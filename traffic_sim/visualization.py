"""Matplotlib rendering: a static network plot and a live animation.

All car positions come from :meth:`RoadNetwork.point_on_edge`; this module
never recomputes geometry itself.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .network import DEFAULT_SPEED_LIMIT


GREEN = "#2ca02c"
YELLOW = "#e8a800"
RED = "#d62728"
ARTERIAL = "#1f77b4"
ELEVATED = "#8c564b"   # elevated highway (overpasses)


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
        from .network import LEVEL_OFFSET

        def pos(n):
            """Screen position of node ``n``, offset diagonally by its level."""
            off = n.level * LEVEL_OFFSET
            return n.x + off, n.y + off

        pairs = {(e.u, e.v) for e in net.edges}
        for e in net.edges:
            n1, n2 = net.nodes[e.u], net.nodes[e.v]
            (x1, y1), (x2, y2) = pos(n1), pos(n2)
            lw = 0.8 + 1.1 * e.lanes
            if n1.level == 1 and n2.level == 1:        # elevated road (overpass)
                # white casing hides the ground road below, then the highway on top.
                ax.plot([x1, x2], [y1, y2], color="white", linewidth=lw + 3.5,
                        solid_capstyle="round", zorder=4)
                ax.plot([x1, x2], [y1, y2], color=ELEVATED, linewidth=lw,
                        solid_capstyle="round", zorder=5)
            elif n1.level != n2.level:                 # ramp between levels
                ax.plot([x1, x2], [y1, y2], color=ELEVATED, linewidth=1.6,
                        linestyle=(0, (2, 1.5)), zorder=4)
            else:                                       # ground street
                arterial = e.speed_limit > DEFAULT_SPEED_LIMIT + 1e-6
                ax.plot([x1, x2], [y1, y2],
                        color=ARTERIAL if arterial else "black",
                        linewidth=lw, solid_capstyle="round", zorder=1)
                one_way = (e.v, e.u) not in pairs
                if arrows or one_way:
                    ax.annotate(
                        "", xytext=(x1, y1),
                        xy=(x1 + 0.6 * (x2 - x1), y1 + 0.6 * (y2 - y1)),
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
        """Green / amber / red colour for each indicator spec at time ``t``.

        ``specs`` is the ``(node_id, orientation, turn)`` list from
        :meth:`_signal_specs`; each entry is coloured by ``signals.state`` for
        that movement (amber during a phase's yellow clearance). Recomputed every
        animation frame.
        """
        from .signals import SignalState

        colors = {SignalState.GREEN: GREEN, SignalState.YELLOW: YELLOW,
                  SignalState.RED: RED}
        return [colors[signals.state(n, o, turn, t)] for (n, o, turn) in specs]

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
            """A square legend swatch of ``color`` labelled ``label``."""
            return Line2D([], [], marker="s", linestyle="None", markersize=8,
                          markerfacecolor=color, markeredgecolor="0.3", label=label)

        handles = [
            square("0.55", "top-left  =  N–S through"),
            square("0.55", "top-right  =  N–S left"),
            square("0.55", "right, lower  =  E–W through"),
            square("0.55", "right, upper  =  E–W left"),
            square(GREEN, "green  =  may go"),
            square(YELLOW, "amber  =  clearing (stop if you can)"),
            square(RED, "red  =  stop"),
        ]
        return ax.legend(handles=handles,
                         title="Signal squares\n(4 per junction, 1 green at a time)",
                         loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         fontsize=8, title_fontsize=8, framealpha=0.95,
                         borderaxespad=0.0)

    def _draw_zone_glow(self, ax, net, zones, *, strength: float = 1.0,
                        zorder: float = 0.6):
        """Draw a soft land-use **glow around each street**; return legend handles.

        The glow is a stack of translucent colour bands *wider* than the road and
        drawn *beneath* it (low ``zorder``), so the street keeps its own colour
        (black local / blue arterial / brown elevated) and just sits in a coloured
        halo — the zone reads from the neighbourhood, not by recolouring the road.
        Two bands (a wide faint outer one + a tighter brighter inner one) fake a
        blur; ``strength`` scales their opacity. ``zones`` is an
        ``{edge_id: LandUse}`` map (only ground streets are zoned). Shared by the
        land-use map, static frames and the live demo for consistent colours.
        Returns the legend handles for the uses present.
        """
        from .zones import LandUse
        from matplotlib.lines import Line2D

        colors = {LandUse.RESIDENTIAL: "#2ca02c", LandUse.OFFICE: "#1f77b4",
                  LandUse.RETAIL: "#ff7f0e", LandUse.OTHER: "#b0b0b0"}
        # (extra line width, opacity) per band — outer halo then inner glow.
        bands = [(11.0, 0.16 * strength), (6.0, 0.30 * strength)]
        for e in net.edges:
            use = zones.get(e.id)
            if use is None:
                continue                               # elevated / unzoned edge
            n1, n2 = net.nodes[e.u], net.nodes[e.v]
            base = 0.8 + 1.1 * e.lanes
            for extra, alpha in bands:
                ax.plot([n1.x, n2.x], [n1.y, n2.y], color=colors[use],
                        linewidth=base + extra, alpha=alpha,
                        solid_capstyle="round", zorder=zorder)
        return [Line2D([], [], color=colors[u], linewidth=6, alpha=0.5, label=u.value)
                for u in LandUse if any(v is u for v in zones.values())]

    def _zone_legend(self, ax, handles):
        """Create a 'land use' legend on the right, stacked *below* the signal one.

        Returns the legend; the caller re-pins the signal legend via
        :meth:`~matplotlib.axes.Axes.add_artist` when both are shown (a second
        ``ax.legend`` call otherwise evicts the first).
        """
        return ax.legend(handles=handles, title="land use", loc="lower left",
                         bbox_to_anchor=(1.02, 0.0), fontsize=8, title_fontsize=8,
                         framealpha=0.95, borderaxespad=0.0)

    def render_state(self, net, cars=None, t: float = 0.0, signals=None,
                     zones=None, path: str = "frame.png", title: str = None):
        """Render a single headless frame to ``path`` (PNG). Returns the path.

        When ``signals`` is given, a legend keys the four per-junction signal
        squares (see :meth:`_signal_legend`); ``bbox_inches="tight"`` keeps it in
        the saved image. Pass ``zones`` (an ``{edge_id: LandUse}`` map) to wash the
        streets in their land-use colour under the traffic — both legends then
        stack on the right.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        if zones is not None:                       # land-use glow around the roads
            zone_handles = self._draw_zone_glow(ax, net, zones)
        else:
            zone_handles = None
        self._draw_edges(ax, net)
        ax.scatter([n.x for n in net.nodes], [n.y for n in net.nodes],
                   color="black", s=6, zorder=2)
        sig_leg = None
        if signals is not None:
            self._draw_signals(ax, net, signals, t)
            sig_leg = self._signal_legend(ax)
        if zone_handles:
            self._zone_legend(ax, zone_handles)
            if sig_leg is not None:                  # re-pin the evicted signal legend
                ax.add_artist(sig_leg)
        if cars:
            xs, ys = zip(*(net.point_on_edge_lane(c.edge_id, c.s, c.lane) for c in cars))
            # Near-black over the coloured land wash (blue would clash with office).
            ax.scatter(xs, ys, s=55, zorder=5,
                       color="#1a1a1a" if zones is not None else "tab:blue")

        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 12, max_x + 12)
        ax.set_ylim(min_y - 12, max_y + 12)
        ax.set_title(title or f"t = {t:.1f}s")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path

    def render_zones(self, net, zones, path: str = "zones.png", title: str = None):
        """Render the land-use map: streets coloured by zone, with a legend.

        ``zones`` is an ``{edge_id: LandUse}`` map (see :mod:`traffic_sim.zones`).
        The streets keep their normal styling; a land-use **glow** around each one
        shows the zoning. Returns the PNG path.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        handles = self._draw_zone_glow(ax, net, zones, strength=1.4)
        self._draw_edges(ax, net)
        ax.legend(handles=handles, title="land use", loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), fontsize=9, borderaxespad=0.0)
        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 12, max_x + 12)
        ax.set_ylim(min_y - 12, max_y + 12)
        ax.set_title(title or "Land-use zones")
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

    def render_fundamental_diagram(self, metrics, net, path="fundamental.png",
                                   title=None, max_points=8000):
        """Plot the emergent fundamental diagram from per-edge samples.

        Two panels over all ``(density, flow, speed)`` samples (one per occupied
        edge per step; requires the collector's ``record_edges=True``): flow vs
        density (coloured by speed) and speed vs density (coloured by flow). The
        free-flow branch, the capacity peak and the congested branch appear
        without being imposed. Returns the PNG path.
        """
        samples = metrics.fundamental_samples(net)
        if not samples:
            raise ValueError("no edge samples — build the collector with record_edges=True")
        if len(samples) > max_points:                       # thin for legibility
            stride = len(samples) // max_points + 1
            samples = samples[::stride]
        k = [s[0] for s in samples]
        q = [s[1] for s in samples]
        v = [s[2] for s in samples]

        fig, (ax_q, ax_v) = plt.subplots(1, 2, figsize=(11, 4.5))
        sc1 = ax_q.scatter(k, q, s=7, alpha=0.35, c=v, cmap="viridis")
        ax_q.set_xlabel("density  k  [veh/km]")
        ax_q.set_ylabel("flow  q = k·v  [veh/h]")
        fig.colorbar(sc1, ax=ax_q, label="speed [km/h]")
        sc2 = ax_v.scatter(k, v, s=7, alpha=0.35, c=q, cmap="magma")
        ax_v.set_xlabel("density  k  [veh/km]")
        ax_v.set_ylabel("speed  v  [km/h]")
        fig.colorbar(sc2, ax=ax_v, label="flow [veh/h]")
        fig.suptitle(title or "Fundamental diagram (emergent)")
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path

    def render_delay_distribution(self, metrics, path="delay_dist.png",
                                  title=None, bins=30):
        """Histogram of per-trip delay with mean and p95 marked. Returns the path."""
        delays = metrics.delays
        if not delays:
            raise ValueError("no completed trips — use a destination-aware router")
        s = sorted(delays)
        mean = sum(s) / len(s)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delays, bins=bins, color="tab:blue", edgecolor="white")
        ax.axvline(mean, color="tab:red", lw=2, label=f"mean {mean:.1f} s")
        ax.axvline(p95, color="tab:orange", lw=2, ls="--", label=f"p95 {p95:.1f} s")
        ax.set_xlabel("per-trip delay [s]")
        ax.set_ylabel("trips")
        ax.set_title(title or f"Trip delay distribution ({len(delays)} trips)")
        ax.legend()
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
            xs, ys = zip(*(net.point_on_edge_lane(c.edge_id, c.s, c.lane) for c in cars))
            ax.scatter(xs, ys, s=50)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Road network")
        plt.show()

    def _build_animation(self, net, sim, dt, steps, zones=None, show_signals=True,
                         day_length=None, fps=None, steps_per_frame=1):
        """Construct the matplotlib ``FuncAnimation`` that drives the sim.

        Sets up the static backdrop (edges, land-use nodes, signal markers), then
        returns ``(fig, anim)`` where each frame advances the sim and moves the car
        scatter / recolours the signals / ticks the clock. Shared by
        :meth:`animate_sim` (live window) and :meth:`save_animation` (GIF), which
        differ only in how they consume the returned animation. ``zones``
        (optional ``{node_id: LandUse}``) colours the nodes by land use.
        ``show_signals=False`` hides the traffic-light markers and legend (the
        signals still operate — this only declutters the view, handy on big maps
        or when assessing zones). ``day_length`` (simulated seconds per day; taken
        from the demand model when omitted) drives a live **wall-clock** overlay
        that runs midnight→midnight over one day.

        **Playback speed.** ``fps`` sets the target frame rate (the inter-frame
        delay is ``1000/fps`` ms; defaults to real-time ``dt``). ``steps_per_frame``
        advances the sim that many steps between redraws — so the day plays that
        many times faster (fewer, cheaper renders per simulated second) at the cost
        of choppier car motion. The sim still runs the full ``steps`` steps; the
        animation just renders ``steps // steps_per_frame`` frames.
        """
        fig, ax = plt.subplots(figsize=(8.6, 6))
        # Land-use glow first (under the streets), so each street shows a soft
        # colour halo without recolouring the black road or the traffic.
        zone_handles = (self._draw_zone_glow(ax, net, zones)
                        if zones is not None else None)
        self._draw_edges(ax, net)
        min_x, min_y, max_x, max_y = net.bounds()
        ax.set_aspect("equal")
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(min_y - 10, max_y + 10)

        signals = sim.signals if show_signals else None
        sig_scat = None
        sig_leg = None
        if signals is not None:
            positions, specs = self._signal_specs(net, signals)
            if positions:
                pos = np.array(positions)
                sig_scat = ax.scatter(pos[:, 0], pos[:, 1], s=16, marker="s", zorder=4)
                sig_leg = self._signal_legend(ax)
        if zone_handles:
            self._zone_legend(ax, zone_handles)
            if sig_leg is not None:                  # re-pin the evicted signal legend
                ax.add_artist(sig_leg)
        if sig_scat is not None or zone_handles:
            # Reserve room on the right so the legend(s) sit beside the map.
            fig.subplots_adjust(left=0.06, right=0.66)

        # Cars in near-black so they stay salient over the coloured land wash
        # (blue would clash with the office zone). Active cars are semi-transparent
        # so overlapping traffic builds up colour density — jams read darker;
        # parked / asleep cars nearly vanish (tiny, very faint) so the map shows
        # moving traffic, not the whole parked fleet.
        car_color = "#1a1a1a" if zones is not None else "tab:blue"
        scat = ax.scatter([], [], s=32, color=car_color, alpha=0.55,
                          edgecolors="none", zorder=5)
        parked = ax.scatter([], [], s=4, color=car_color, alpha=0.12,
                            edgecolors="none", zorder=3)

        def _offsets(cars):
            """Screen positions for ``cars`` as an (N, 2) array (empty-safe)."""
            pts = [net.point_on_edge_lane(c.edge_id, c.s, c.lane) for c in cars]
            return np.array(pts) if pts else np.empty((0, 2))

        # Live wall clock: map sim time onto a midnight→midnight day.
        if day_length is None:
            demand = getattr(getattr(sim, "router", None), "demand", None)
            day_length = getattr(demand, "day_length", None)
        clock = None
        if day_length:
            clock = ax.text(0.015, 0.985, "", transform=ax.transAxes, ha="left",
                            va="top", fontsize=13, fontweight="bold", zorder=6,
                            family="monospace",
                            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

        def clock_label(t):
            """``HH:MM  ·  Period`` for sim time ``t`` on the midnight-based day."""
            frac = (t % day_length) / day_length
            minutes = int(round(frac * 1440)) % 1440
            # Quarters match DemandModel.period (Night, Morning, Midday, Evening).
            name = ("Night", "Morning", "Midday", "Evening")[min(3, int(frac * 4))]
            return f"{minutes // 60:02d}:{minutes % 60:02d}  ·  {name}"

        def dynamic():
            """The blit artists that change each frame (skip the absent ones)."""
            return tuple(a for a in (scat, parked, sig_scat, clock) if a is not None)

        def init():
            """Blit initialiser: empty car scatters, clock at the start of the day."""
            scat.set_offsets(np.empty((0, 2)))
            parked.set_offsets(np.empty((0, 2)))
            if clock is not None:
                clock.set_text(clock_label(sim.t))
            return dynamic()

        def update(frame):
            """Advance the sim ``steps_per_frame`` steps and redraw cars/signals/clock."""
            for _ in range(steps_per_frame):
                sim.step(dt)
            scat.set_offsets(_offsets([c for c in sim.cars if c.active]))
            parked.set_offsets(_offsets([c for c in sim.cars if not c.active]))
            if sig_scat is not None:
                sig_scat.set_color(self._signal_colors(signals, sim.t, specs))
            if clock is not None:
                clock.set_text(clock_label(sim.t))
            return dynamic()

        interval = 1000.0 / fps if fps else dt * 1000
        n_frames = max(1, steps // max(1, steps_per_frame))
        anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                             interval=interval, blit=True, repeat=False)
        return fig, anim

    def animate_sim(self, net, sim, dt: float = 0.1, steps: int = 400, zones=None,
                    show_signals: bool = True, day_length=None, fps: int = 30,
                    steps_per_frame: int = 1):
        """Open a live animation window (signal colours update each frame).

        Pass ``zones`` (a ``{node_id: LandUse}`` map) to colour the nodes by land
        use so the chosen zoning is visible under the traffic. ``show_signals=False``
        hides the traffic-light markers to declutter big maps (signals still run).
        ``day_length`` drives the live wall-clock overlay. ``fps`` sets the target
        frame rate and ``steps_per_frame`` plays the day that many times faster
        (see :meth:`_build_animation`).
        """
        _, anim = self._build_animation(net, sim, dt, steps, zones=zones,
                                        show_signals=show_signals, day_length=day_length,
                                        fps=fps, steps_per_frame=steps_per_frame)
        plt.show()
        return anim

    def save_animation(self, net, sim, path: str, dt: float = 0.1, steps: int = 400,
                       fps: int = 20, zones=None, show_signals: bool = True,
                       day_length=None, steps_per_frame: int = 1):
        """Render the simulation to a GIF at ``path`` (headless, no window).

        ``zones`` optionally colours the nodes by land use (see :meth:`animate_sim`);
        ``show_signals=False`` hides the traffic-light markers; ``day_length`` drives
        the wall-clock overlay; ``fps`` is the GIF frame rate and ``steps_per_frame``
        compresses the day (fewer frames).
        """
        from matplotlib.animation import PillowWriter

        fig, anim = self._build_animation(net, sim, dt, steps, zones=zones,
                                          show_signals=show_signals, day_length=day_length,
                                          fps=fps, steps_per_frame=steps_per_frame)
        anim.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return path
