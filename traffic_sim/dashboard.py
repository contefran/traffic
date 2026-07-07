"""Interactive dashboard: the live demo plus tuning knobs and live metrics.

Wraps the live animation in a control panel built from matplotlib's own
widgets (no new dependency). A handful of **sliders** mutate the *running*
simulation in place — street speed limits, following distances, driver
acceleration, signal timing, playback speed — and a **metrics panel** shows
the flow statistics collected since the last press of its *Reset* button
(:meth:`MetricsCollector.reset`). The workflow it supports: turn a knob,
press Reset, and watch the statistics of the new regime accumulate cleanly.

The knobs deliberately tune the *traffic*, never the city layout — speed
policy, driver behaviour and signal timing are exactly the levers a traffic
engineer (or, later, a learned controller) can pull.

Simplifications, by design:

* Speed-limit knobs change the dynamics immediately, but the router's cached
  free-flow cost tables were built once — cars keep choosing routes as if the
  original limits held (and trip-delay baselines drift accordingly). Fine for
  live exploration; rebuild the router for a rigorous experiment.
* The dashboard runs **day after day** (the demand and schedules are
  periodic), ignoring the demo's ``--steps``; close the window to end the run
  and print the summary (which then covers the window since the last Reset).
* The legends are dropped to make room for the panel — the plain demo keeps
  them.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from .network import DEFAULT_SPEED_LIMIT
from .signals import SignalPlan
from .units import ms_to_kmh
from .visualization import Visuals


class Dashboard(Visuals):
    """The live demo with tuning knobs and a resettable live-metrics panel.

    Reuses the :class:`Visuals` drawing helpers for the map itself; everything
    on the right-hand side (clock, metrics panel, Reset button, sliders) lives
    in its own little axes so it survives blitting. Build one with the same
    ingredients as the demo animation and call :meth:`show`::

        Dashboard(net, sim, zones, day_length=1200.0).show()

    Knob **baselines are captured at construction** (per-edge speed limits,
    per-driver gaps/acceleration), so a multiplier slider always scales from
    the original values — dragging it twice never compounds, and the per-driver
    jitter of the fleet is preserved under scaling.
    """

    def __init__(self, net, sim, zones=None, *, dt: float = 0.1,
                 day_length=None, show_signals: bool = True, fps: int = 30,
                 steps_per_frame: int = 1):
        self.net, self.sim, self.zones = net, sim, zones
        self.dt = dt
        self.day_length = self._resolve_day_length(sim, day_length)
        self.show_signals = show_signals
        self.fps = fps
        self._spf = max(1, steps_per_frame)     # live "sim speed" knob value
        # Baselines the multiplier knobs scale from (captured once).
        self._base_speed = {e.id: e.speed_limit for e in net.edges}
        self._base_driver = {c.id: (c.time_headway, c.s0, c.accel)
                             for c in sim.cars}
        self._local_edges, self._fast_edges = self._classify_edges(net)
        self.sliders = {}
        self._widgets = []          # keep refs — a GC'd widget loses its callbacks

    # ------------------------------------------------------------------ knobs

    @staticmethod
    def _classify_edges(net):
        """Split edge ids into (local streets, fast roads) for the speed knobs.

        Fast roads are the arterials and everything elevated (highway + ramps);
        local is the rest of the ground grid, including residential streets
        (which keep their *relative* 30 km/h calm under scaling). Roundabout
        ring edges are skipped entirely — their low speed is part of the
        geometry, not a policy knob.
        """
        local, fast = [], []
        for e in net.edges:
            u, v = net.nodes[e.u], net.nodes[e.v]
            if u.internal or v.internal:
                continue
            if (u.level > 0 or v.level > 0
                    or e.speed_limit > DEFAULT_SPEED_LIMIT + 1e-6):
                fast.append(e.id)
            else:
                local.append(e.id)
        return local, fast

    def _scale_speeds(self, edge_ids, factor):
        """Set each edge's speed limit to ``factor`` times its original value."""
        for eid in edge_ids:
            self.net.edges[eid].speed_limit = self._base_speed[eid] * factor

    def _scale_following(self, factor):
        """Scale every driver's following distance (time headway *and*
        standstill gap) from its original — jittered — value."""
        for c in self.sim.cars:
            th, s0, _ = self._base_driver[c.id]
            c.time_headway, c.s0 = th * factor, s0 * factor

    def _scale_accel(self, factor):
        """Scale every driver's comfortable acceleration from its original value."""
        for c in self.sim.cars:
            c.accel = self._base_driver[c.id][2] * factor

    def _signal_controller(self):
        """The plan-carrying signal controller, or ``None`` (no signals /
        a controller without :class:`SignalPlan` timing)."""
        controller = getattr(getattr(self.sim, "signals", None), "controller", None)
        return controller if hasattr(controller, "default_plan") else None

    def _retime_signals(self, green: float, yellow: float) -> None:
        """Re-time every intersection: ``green`` per phase, ``yellow`` clearance.

        Rebuilds the controller's default plan and any per-node plans with the
        new durations, keeping each node's offset (phases jump at the moment of
        change — a one-off transient, like re-programming real signals).
        """
        controller = self._signal_controller()
        if controller is None:
            return

        def retimed(plan):
            return SignalPlan((green,) * len(plan.green_times),
                              offset=plan.offset, yellow=yellow)

        controller.default_plan = retimed(controller.default_plan)
        controller.plans = {n: retimed(p) for n, p in controller.plans.items()}

    # ----------------------------------------------------------------- panel

    def _panel_text(self) -> str:
        """The metrics panel body: flow statistics since the last Reset."""
        m = getattr(self.sim, "metrics", None)
        if m is None:
            return "no metrics collector attached"
        if not m.history:
            return "collecting…"
        n = len(m.history)
        last = m.history[-1]
        active = sum(1 for c in self.sim.cars if getattr(c, "active", True))
        lines = [
            f"window       {m.times[-1] - m.times[0]:8.0f} s",
            f"active cars  {active:8d}",
            f"speed now    {ms_to_kmh(last.mean_speed):8.1f} km/h",
            f"speed avg    {ms_to_kmh(sum(m.mean_speeds) / n):8.1f} km/h",
            f"queue now    {last.n_stopped:8d}",
            f"queue avg    {sum(m.queue_lengths) / n:8.1f}",
            f"trips done   {len(m.trips):8d}",
        ]
        if m.trips:
            lines.append(f"mean delay   {sum(m.delays) / len(m.trips):8.1f} s")
        lines += [
            f"crashes      {m.crashes:8d}",
            f"fuel proxy   {m.fuel_proxy:8.0f}",
        ]
        return "\n".join(lines)

    def _on_reset(self, _event=None) -> None:
        """Reset button: start a fresh metrics window (the sim runs on)."""
        m = getattr(self.sim, "metrics", None)
        if m is not None:
            m.reset()

    # ----------------------------------------------------------------- build

    def _add_slider(self, fig, y, key, label, lo, hi, init, on_change,
                    valstep=None):
        """One knob row: a slider at height ``y``, wired to ``on_change``."""
        ax = fig.add_axes([0.70, y, 0.24, 0.028])
        slider = Slider(ax, label, lo, hi, valinit=init, valstep=valstep)
        slider.on_changed(on_change)
        self.sliders[key] = slider
        self._widgets.append(slider)
        return slider

    def build(self):
        """Construct the dashboard figure (map, panel, knobs); return it.

        Split from :meth:`show` so tests can build and poke the widgets
        headlessly. The animation is stored on ``self`` (it must outlive this
        call — a garbage-collected ``FuncAnimation`` silently stops).
        """
        fig = plt.figure(figsize=(12.6, 7.2))
        ax = fig.add_axes([0.01, 0.02, 0.55, 0.96])
        self._map_backdrop(ax, self.net, self.zones)

        signals = self.sim.signals if self.show_signals else None
        sig_scat, specs = None, None
        if signals is not None:
            positions, specs = self._signal_specs(self.net, signals)
            if positions:
                pos = np.array(positions)
                sig_scat = ax.scatter(pos[:, 0], pos[:, 1], s=16, marker="s",
                                      zorder=4)

        clock = (self._make_clock(fig, [0.58, 0.93, 0.40, 0.05])
                 if self.day_length else None)

        # Metrics panel: monospace text in its own axes (blit-safe), with the
        # Reset button in its top-right corner.
        panel_ax = fig.add_axes([0.60, 0.52, 0.36, 0.38])
        panel_ax.axis("off")
        panel_ax.set_title("metrics — since reset", loc="left", fontsize=10)
        panel = panel_ax.text(0.02, 0.96, "collecting…",
                              transform=panel_ax.transAxes, ha="left", va="top",
                              fontsize=11, family="monospace")
        reset_btn = Button(fig.add_axes([0.87, 0.865, 0.09, 0.042]), "Reset")
        reset_btn.on_clicked(self._on_reset)
        self._widgets.append(reset_btn)

        # The knobs. Multipliers scale from construction-time baselines; the
        # signal knobs re-time every plan; "sim speed" only changes playback.
        ys = itertools.count()
        y = lambda: 0.44 - 0.055 * next(ys)         # noqa: E731 — knob rows
        self._add_slider(fig, y(), "local", "local streets ×", 0.4, 1.6, 1.0,
                         lambda v: self._scale_speeds(self._local_edges, v))
        self._add_slider(fig, y(), "fast", "fast roads ×", 0.4, 1.6, 1.0,
                         lambda v: self._scale_speeds(self._fast_edges, v))
        self._add_slider(fig, y(), "follow", "following gap ×", 0.5, 2.2, 1.0,
                         self._scale_following)
        self._add_slider(fig, y(), "accel", "acceleration ×", 0.5, 1.6, 1.0,
                         self._scale_accel)
        controller = self._signal_controller()
        if controller is not None:
            plan = controller.default_plan
            retime = lambda _v: self._retime_signals(     # noqa: E731
                self.sliders["green"].val, self.sliders["yellow"].val)
            self._add_slider(fig, y(), "green", "green time [s]", 2.0, 15.0,
                             plan.green_times[0], retime)
            self._add_slider(fig, y(), "yellow", "yellow [s]", 0.0, 4.0,
                             plan.yellow, retime)
        self._add_slider(fig, y(), "speed", "sim speed [steps/frame]", 1, 10,
                         self._spf, lambda v: setattr(self, "_spf", int(v)),
                         valstep=1)

        # Car layers, as in the demo: semi-transparent active cars (jams read
        # darker), near-invisible parked ones, marker size by vehicle length.
        car_color = "#1a1a1a" if self.zones is not None else "tab:blue"
        scat = ax.scatter([], [], s=32, color=car_color, alpha=0.55,
                          edgecolors="none", zorder=5)
        parked = ax.scatter([], [], s=4, color=car_color, alpha=0.12,
                            edgecolors="none", zorder=3)

        def dynamic():
            """The blit artists that change each frame (skip the absent ones)."""
            return tuple(a for a in (scat, parked, sig_scat, clock, panel)
                         if a is not None)

        def update(_frame):
            """Advance the sim ``_spf`` steps, redraw cars/signals/clock/panel."""
            for _ in range(self._spf):
                self.sim.step(self.dt)
            active = [c for c in self.sim.cars if c.active]
            scat.set_offsets(self._offsets(self.net, active))
            scat.set_sizes([14.0 + 3.5 * c.length for c in active])
            parked.set_offsets(self._offsets(
                self.net, [c for c in self.sim.cars if not c.active]))
            if sig_scat is not None:
                sig_scat.set_color(self._signal_colors(signals, self.sim.t, specs))
            if clock is not None:
                clock.set_text(self._clock_label(self.sim.t, self.day_length))
            panel.set_text(self._panel_text())
            return dynamic()

        self._update = update       # exposed for headless tests
        # Runs day after day (frames is unbounded); closing the window stops it.
        self._anim = FuncAnimation(fig, update, frames=itertools.count(),
                                   interval=1000.0 / self.fps, blit=True,
                                   repeat=False, cache_frame_data=False)
        return fig

    def show(self) -> None:
        """Open the dashboard window and block until it is closed."""
        self.build()
        plt.show()
