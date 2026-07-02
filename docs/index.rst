Traffic Simulator
=================

An explicit, step-based 2D traffic simulation in Python. The project favours
readable code and easy extensibility over physical realism — every behaviour is
meant to be understandable by reading a few functions.

The simulation core (network, vehicles, simulation loop, routing, signals,
right-of-way, metrics) has **no plotting dependency**; only the optional
:class:`~traffic_sim.visualization.Visuals` helper pulls in matplotlib. That
decoupling keeps the core headless-friendly for the planned serving path.

Feature highlights
------------------

* Directed road graph: a uniform grid or a heterogeneous "city" grid with
  jittered positions, one-way streets, dropped links and higher-speed arterials
  (always repaired to stay strongly connected).
* Car-following via the **Intelligent Driver Model** — smooth acceleration and
  braking, realistic queues, and a density-dependent speed that *emerges*.
* **Decentralized** traffic signals: each intersection runs its own
  cycle / split / offset (no global clock), with fixed-time and protected-phase
  controllers.
* Right-of-way (arterial priority + gap acceptance) at unsignalized nodes.
* Destination-based **fastest-path** routing, plus random wandering.
* Flow metrics (speed, queue length, throughput) and 2D animation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
