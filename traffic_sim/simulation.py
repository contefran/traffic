"""Step-based traffic simulation.

The whole dynamics live in :meth:`TrafficSim.step`: cars are grouped per edge,
sorted front-to-back, advanced with the Intelligent Driver Model (IDM), and
handed to the router when they reach the end of an edge. A red traffic light is
modelled as a stationary virtual obstacle at the stop line, so the same model
brakes for it. Density-dependent speed (the fundamental diagram) emerges from
IDM rather than being imposed.
"""

import math
from typing import Dict, List, Optional, Tuple

from .network import RoadNetwork
from .vehicles import Car
from .routing import RandomRouter
from .signals import SignalSystem
from .priority import PriorityModel

# Hard safety buffer kept between a follower and its leader [m].
LEADER_BUFFER = 0.1
# IDM free-acceleration exponent (standard value).
IDM_DELTA = 4.0


class TrafficSim:
    def __init__(
        self,
        net: RoadNetwork,
        cars: List[Car],
        router: Optional[RandomRouter] = None,
        signals: Optional[SignalSystem] = None,
        priority: Optional[PriorityModel] = None,
        metrics=None,
    ) -> None:
        self.net = net
        self.cars = cars
        self.router = router if router is not None else RandomRouter(net)
        self.signals = signals  # None => no signals, every approach is green
        # Right-of-way at unsignalized nodes; None => no yielding (free-for-all).
        self.priority = priority
        self.metrics = metrics  # optional MetricsCollector; observes each step
        self.t = 0.0

    def _unsignalized(self, node_id: int) -> bool:
        return self.signals is None or not self.signals.is_signalized(node_id)

    def _approach_fronts(self, cars_on_edge: Dict[int, List[Car]]) -> Dict[int, list]:
        """Per unsignalized node, the front car of each approach that is near
        enough to contest, as ``(from_edge, to_edge, gap, speed)``. Commits each
        such car's ``next_edge`` so its intended movement is known.
        """
        fronts: Dict[int, list] = {}
        for edge_id, lst in cars_on_edge.items():
            edge = self.net.edges[edge_id]
            if not self._unsignalized(edge.v):
                continue
            front = lst[0]  # lst is sorted front (high s) -> back
            gap = edge.length - front.s
            if gap > self.priority.trigger_dist:
                continue
            if front.next_edge is None:
                front.next_edge = self.router.next_edge(edge_id, front)
            fronts.setdefault(edge.v, []).append(
                (edge_id, front.next_edge, gap, front.v))
        return fronts

    def _idm_accel(self, car: Car, v_des: float,
                   obstacle: Optional[Tuple[float, float]]) -> float:
        """Intelligent Driver Model acceleration.

        ``obstacle`` is ``(gap, lead_speed)`` for the constraint ahead — a real
        leader or a red stop line (a stationary obstacle, ``lead_speed=0``) — or
        ``None`` for open road. The free term accelerates toward ``v_des``; the
        interaction term brakes for the obstacle. Reuses the car's existing
        parameters: ``accel`` (a), ``braking`` (b), ``s0``, ``time_headway`` (T).
        """
        free = 1.0 - (car.v / v_des) ** IDM_DELTA if v_des > 0 else 0.0
        if obstacle is None:
            return car.accel * free

        gap, lead_v = obstacle
        approach_rate = car.v - lead_v  # > 0 when closing on the obstacle
        s_star = car.s0 + max(
            0.0,
            car.v * car.time_headway
            + (car.v * approach_rate) / (2.0 * math.sqrt(car.accel * car.braking)),
        )
        gap = max(gap, 0.01)  # guard against division blow-up at zero gap
        return car.accel * (free - (s_star / gap) ** 2)

    def step(self, dt: float) -> None:
        # Group cars by edge and sort each edge front (high s) -> back.
        cars_on_edge: Dict[int, List[Car]] = {}
        for car in self.cars:
            cars_on_edge.setdefault(car.edge_id, []).append(car)
        for lst in cars_on_edge.values():
            lst.sort(key=lambda c: c.s, reverse=True)

        # Right-of-way contest data at unsignalized nodes (empty if disabled).
        fronts = self._approach_fronts(cars_on_edge) if self.priority is not None else {}

        # Defer edge transfers so a car moving to a new edge does not disturb
        # the leader/follower ordering of the edge currently being processed.
        transfers: List[tuple] = []  # (car, next_edge_id, new_s)

        for edge_id, lst in cars_on_edge.items():
            edge = self.net.edges[edge_id]

            for idx, car in enumerate(lst):
                leader = lst[idx - 1] if idx > 0 else None
                v_des = min(edge.speed_limit, car.max_speed)

                # Commit the next edge in advance so the signal can gate this
                # car's specific movement (e.g. a protected left vs a through).
                if car.next_edge is None:
                    car.next_edge = self.router.next_edge(edge_id, car)

                red = (
                    self.signals is not None
                    and car.next_edge is not None
                    and not self.signals.allows_movement(edge_id, car.next_edge, self.t)
                )

                # At an unsignalized node the front car of an approach may have
                # to yield right-of-way to conflicting higher-priority traffic.
                if (self.priority is not None and idx == 0
                        and self._unsignalized(edge.v)
                        and self.priority.must_yield(edge_id, car.next_edge,
                                                     fronts.get(edge.v, []))):
                    red = True

                # Constraints ahead, each (gap, speed): the leader and/or, on
                # red, the stop line at the end of the edge (a stopped object).
                obstacles = []
                if leader is not None:
                    obstacles.append((leader.s - car.s - car.length, leader.v))
                if red:
                    obstacles.append((edge.length - car.s, 0.0))

                # Most restrictive (smallest) IDM acceleration over obstacles.
                if obstacles:
                    a = min(self._idm_accel(car, v_des, o) for o in obstacles)
                else:
                    a = self._idm_accel(car, v_des, None)

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

                # Reached the end of the edge.
                if new_s >= edge.length:
                    if car.next_edge is None:
                        new_s = edge.length  # dead-end: clamp and stop
                        car.v = 0.0
                    elif not red:
                        overshoot = new_s - edge.length
                        next_len = self.net.edges[car.next_edge].length
                        transfers.append((car, car.next_edge, min(overshoot, next_len)))
                        continue  # applied in the transfer pass
                    # If red, fall through: the car waits at the stop line.

                car.s = new_s
                car.trail.append((self.t, car.edge_id, car.s))

        for car, next_edge, new_s in transfers:
            car.edge_id = next_edge
            car.next_edge = None  # re-route from the new edge next step
            car.s = new_s
            car.trail.append((self.t, car.edge_id, car.s))

        self.t += dt

        if self.metrics is not None:
            self.metrics.record(self)
