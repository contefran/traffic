"""Step-based traffic simulation.

The whole dynamics live in :meth:`TrafficSim.step`: cars are grouped per edge,
sorted front-to-back, advanced with a simple car-following rule, and handed to
the router when they reach the end of an edge. A red traffic light is modelled
as a virtual obstacle at the stop line, so the same car-following rule makes a
car brake smoothly for it.
"""

from typing import Dict, List, Optional

from .network import RoadNetwork
from .vehicles import Car
from .routing import RandomRouter
from .signals import SignalSystem

# Hard safety buffer kept between a follower and its leader [m].
LEADER_BUFFER = 0.1


class TrafficSim:
    def __init__(
        self,
        net: RoadNetwork,
        cars: List[Car],
        router: Optional[RandomRouter] = None,
        signals: Optional[SignalSystem] = None,
    ) -> None:
        self.net = net
        self.cars = cars
        self.router = router if router is not None else RandomRouter(net)
        self.signals = signals  # None => no signals, every approach is green
        self.t = 0.0

    def _desired_accel(self, car: Car, gap: Optional[float], v_des: float) -> float:
        """Car-following acceleration: brake if the gap ahead is unsafe, else cruise.

        ``gap`` is the clear distance to whatever limits the car (a leader or a
        red stop line), or ``None`` if the road ahead is open.
        """
        if gap is not None:
            safe = car.s0 + car.time_headway * car.v
            if gap < safe:
                return -car.braking
        if car.v < v_des:
            return car.accel
        return 0.0

    def step(self, dt: float) -> None:
        # Group cars by edge and sort each edge front (high s) -> back.
        cars_on_edge: Dict[int, List[Car]] = {}
        for car in self.cars:
            cars_on_edge.setdefault(car.edge_id, []).append(car)
        for lst in cars_on_edge.values():
            lst.sort(key=lambda c: c.s, reverse=True)

        # Defer edge transfers so a car moving to a new edge does not disturb
        # the leader/follower ordering of the edge currently being processed.
        transfers: List[tuple] = []  # (car, next_edge_id, new_s)

        for edge_id, lst in cars_on_edge.items():
            edge = self.net.edges[edge_id]
            red = self.signals is not None and not self.signals.is_green(edge_id, self.t)

            for idx, car in enumerate(lst):
                leader = lst[idx - 1] if idx > 0 else None
                v_des = min(edge.speed_limit, car.max_speed)

                # Distance to the nearest constraint ahead: the leader and/or,
                # on red, the stop line at the end of the edge.
                gaps = []
                if leader is not None:
                    gaps.append(leader.s - car.s - car.length)
                if red:
                    gaps.append(edge.length - car.s)
                gap = min(gaps) if gaps else None

                a = self._desired_accel(car, gap, v_des)
                car.v = max(0.0, min(v_des, car.v + a * dt))
                new_s = car.s + car.v * dt

                # Backstop: never overshoot the leader or cross a red stop line.
                max_s = None
                if leader is not None:
                    max_s = leader.s - car.length - LEADER_BUFFER
                if red:
                    max_s = edge.length if max_s is None else min(max_s, edge.length)
                if max_s is not None and new_s > max_s:
                    new_s = max(car.s, max_s)
                    car.v = 0.0

                # On green, hand over to the router at the end of the edge.
                if not red and new_s >= edge.length:
                    next_edge = self.router.next_edge(edge_id)
                    if next_edge is None:
                        # Dead-end: clamp and stop.
                        new_s = edge.length
                        car.v = 0.0
                    else:
                        overshoot = new_s - edge.length
                        next_len = self.net.edges[next_edge].length
                        transfers.append((car, next_edge, min(overshoot, next_len)))
                        continue  # s/edge applied during the transfer pass

                car.s = new_s
                car.trail.append((self.t, car.edge_id, car.s))

        for car, next_edge, new_s in transfers:
            car.edge_id = next_edge
            car.s = new_s
            car.trail.append((self.t, car.edge_id, car.s))

        self.t += dt
