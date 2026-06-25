"""Traffic Simulator — an explicit, step-based 2D traffic simulation.

The simulation core (network, vehicles, simulation, routing) has no plotting
dependencies. ``Visuals`` is imported lazily so the core can run headless
(e.g. behind an API) without matplotlib installed.
"""

from .network import Node, Edge, RoadNetwork, build_grid_network
from .vehicles import Car
from .simulation import TrafficSim
from .routing import RandomRouter

__all__ = [
    "Node",
    "Edge",
    "RoadNetwork",
    "build_grid_network",
    "Car",
    "TrafficSim",
    "RandomRouter",
    "Visuals",
]


def __getattr__(name):
    # Lazy access: `from traffic_sim import Visuals` only pulls in matplotlib
    # when actually requested.
    if name == "Visuals":
        from .visualization import Visuals
        return Visuals
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
