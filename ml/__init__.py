"""Machine-learning pipeline on top of the simulator (dataset -> model -> serving).

Unlike ``traffic_sim`` (which stays dependency-free), this package may use
numpy and, later, ML libraries — it consumes the simulator, it is not part
of it.
"""
