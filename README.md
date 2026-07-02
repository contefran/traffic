# Traffic Simulator — traffic simulation in Python

This project is an explicit traffic simulation written in Python.
The goal is clarity and extensibility, not realism at all costs.

The simulator is built incrementally, starting from a road network and cars moving along edges, and progressively adding intersections, traffic lights, routing, metrics, and eventually learning-based agents.

## Project goals

- Simple, explicit data structures (no black boxes)
- Deterministic, step-based simulation
- Visual debugging via 2D animation
- Easy to extend toward intersections, signals, and agent decision-making
- Suitable as a base for later ML / RL experiments

## Current features

- Directed road network: a uniform grid, or a heterogeneous "city" grid with
  jittered positions, one-way streets, missing links, and higher-speed
  arterials (always repaired to stay strongly connected)
- Routing: random wandering, or destination-based fastest-path routing (cars
  steer toward a destination node, preferring faster arterials)
- Nodes and edges with geometry and per-edge speed limits
- Cars following the Intelligent Driver Model (IDM):
  - smooth acceleration and braking
  - realistic queues; density-dependent speed emerges (the fundamental diagram)
  - no rear-end collisions
- Fixed timestep simulation loop
- Flow metrics (speed, queue length, throughput) and 2D animation via matplotlib

Cars cross intersections (a router picks the next edge) and obey traffic
lights: each intersection runs a fixed-time signal, and cars queue at red and
release on green. Signal timing is driven by a pluggable controller, ready for
adaptive or learned policies later. Unsignalized intersections use an optional
right-of-way model (arterial priority + gap acceptance) so minor streets yield
to major-road traffic instead of driving straight through it.

## Roadmap (high level)

1. Cars on edges ✅
2. Intersections + random routing ✅
3. Traffic lights and intersection controllers ✅ (fixed-time and protected-phase)
4. Metrics and diagnostics ✅
5. Destination-based routing ✅ (fastest-path; random wandering still available)
6. ML / RL decision policies

## Project structure

.
├── traffic_sim/          # simulation package
│   ├── network.py        # Node/Edge/RoadNetwork + grid/city builders + geometry
│   ├── vehicles.py       # Car model
│   ├── simulation.py     # TrafficSim step loop (car-following + transfers)
│   ├── routing.py        # RandomRouter + ShortestPathRouter (intersection decisions)
│   ├── signals.py        # traffic lights: controller interface + fixed-time
│   ├── priority.py       # right-of-way / gap acceptance at unsignalized nodes
│   ├── metrics.py        # flow diagnostics: speed, queue, throughput
│   └── visualization.py  # matplotlib plotting, animation, GIF export
├── tests/                # pytest suite
├── main.py               # entry point / example usage
└── requirements.txt

## Requirements

- Python 3.10+
- numpy, matplotlib (visualization); pytest (tests)

Install dependencies with:

pip install -r requirements.txt

## Running the simulation

From the project directory:

python main.py

This will:
- build a small grid road network
- place a few cars on selected edges
- run and animate the simulation

## Running the tests

MPLBACKEND=Agg python -m pytest -q

The simulation core has no plotting dependency, so the tests run without a display.

## Design philosophy

This project deliberately avoids:
- overly detailed vehicle dynamics
- premature optimisation
- large external frameworks

Instead, it prioritises:
- readable code
- explicit state updates
- ease of experimentation

The intent is that every behaviour in the simulation can be understood by reading a few functions.

## Status

This is an active work in progress.
The API and internal structure are expected to evolve as intersections and control logic are added.

## License

MIT
