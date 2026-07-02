"""Traffic Simulator — an explicit, step-based 2D traffic simulation.

The simulation core (network, vehicles, simulation, routing) has no plotting
dependencies. ``Visuals`` is imported lazily so the core can run headless
(e.g. behind an API) without matplotlib installed.
"""

from .network import Node, Edge, RoadNetwork, build_grid_network, build_city_grid
from .vehicles import Car
from .simulation import TrafficSim
from .routing import RandomRouter, ShortestPathRouter
from .signals import (
    Orientation,
    TurnType,
    SignalPlan,
    FixedTimeController,
    ProtectedPhaseController,
    SignalSystem,
)
from .priority import PriorityModel
from .metrics import MetricsCollector, StepMetrics

__all__ = [
    "Node",
    "Edge",
    "RoadNetwork",
    "build_grid_network",
    "build_city_grid",
    "Car",
    "TrafficSim",
    "RandomRouter",
    "ShortestPathRouter",
    "Orientation",
    "TurnType",
    "SignalPlan",
    "FixedTimeController",
    "ProtectedPhaseController",
    "SignalSystem",
    "PriorityModel",
    "MetricsCollector",
    "StepMetrics",
    "Visuals",
]


def __getattr__(name):
    """Lazily resolve :class:`Visuals` on first access.

    Keeps ``import traffic_sim`` free of matplotlib/numpy: those are only pulled
    in when ``traffic_sim.Visuals`` is actually requested, so the simulation core
    stays headless-friendly (e.g. behind an API). Any other attribute raises
    :class:`AttributeError` as usual.
    """
    if name == "Visuals":
        from .visualization import Visuals
        return Visuals
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
