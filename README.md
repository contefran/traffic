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

## Current features (Phase 1)

- Directed road network (grid-based builder)
- Nodes and edges with geometry and speed limits
- Cars moving along edges with:
  - acceleration and braking
  - car-following rule
  - collision avoidance
- Fixed timestep simulation loop
- 2D visualization and animation using matplotlib

Cars currently stop at the end of an edge. Intersections and routing are the next steps.

## Roadmap (high level)

1. Cars on edges ✅
2. Intersections + random routing
3. Traffic lights and intersection controllers
4. Metrics and diagnostics
5. Destination-based routing
6. ML / RL decision policies

## Project structure

.
├── builder.py    # Road network construction (grid builder)
├── traffic.py    # Car model and traffic simulation logic
├── visuals.py    # Plotting and animation
└── main.py       # Entry point / example usage

## Requirements

- Python 3.10+
- numpy
- matplotlib

Install dependencies with:

pip install numpy matplotlib

## Running the simulation

From the project directory:

python main.py

This will:
- build a small grid road network
- place a few cars on selected edges
- run and animate the simulation

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
