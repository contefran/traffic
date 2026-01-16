from collections import deque
from dataclasses import dataclass, field


@dataclass
class Car:
    id: int
    edge_id: int
    s: float # position along edge [0, length)
    v: float # speed [m/s]
    max_speed: float = 50 # [m/s]
    trail: deque = field(default_factory=lambda: deque(maxlen=200))  # (t, s)
    length = 4.5 # [m]
    accel = 2.0 # [m/s^2]
    braking = 4.0 # [m/s^2]
    s0 = 2.0 # minimum gap [m]
    time_headway = 1.2 # desired time headway [s]

    def car_xy(self, net, car):
        e = net.edges[car.edge_id]
        n1 = net.nodes[e.u]
        n2 = net.nodes[e.v]
        t = car.s / e.length if e.length > 0 else 0.0
        x = n1.x + t * (n2.x - n1.x)
        y = n1.y + t * (n2.y - n1.y)
        return x, y
    

class TrafficSim:
    def __init__(self, net, cars):
        self.net = net
        self.cars = cars
        self.t = 0.0

    def step(self, dt: float):
            # group cars by edge
            cars_on_edge = {}
            for car in self.cars:
                cars_on_edge.setdefault(car.edge_id, []).append(car)

            # sort each edge front->back
            for edge_id, lst in cars_on_edge.items():
                lst.sort(key=lambda c: c.s, reverse=True)

            # update cars per edge
            for edge_id, lst in cars_on_edge.items():
                edge = self.net.edges[edge_id]

                for idx, car in enumerate(lst):
                    leader = lst[idx - 1] if idx > 0 else None
                    v_des = min(edge.speed_limit, car.max_speed)

                    a = 0.0
                    if leader is None:
                        if car.v < v_des:
                            a = car.accel
                    else:
                        gap = leader.s - car.s - car.length
                        safe = car.s0 + car.time_headway * car.v

                        if gap < safe:
                            a = -car.braking
                        elif car.v < v_des:
                            a = car.accel

                    # integrate
                    car.v = max(0.0, min(v_des, car.v + a * dt))
                    new_s = car.s + car.v * dt

                    # hard backstop: never pass leader
                    if leader is not None:
                        max_s = leader.s - car.length - 0.1
                        if new_s > max_s:
                            new_s = max(car.s, max_s)
                            car.v = 0.0

                    # stop at end of edge for now (until intersections exist)
                    if new_s > edge.length:
                        new_s = edge.length
                        car.v = 0.0

                    car.s = new_s

                    # history (t, s) is enough; x,y can be derived for plotting
                    car.trail.append((self.t, car.s))

            self.t += dt