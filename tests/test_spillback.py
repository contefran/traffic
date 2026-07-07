"""Spillback invariant: a car is never injected overlapping another.

A persistent overlap re-trips the crash counter every step (a car with
``s > max_s`` can never clear it), so cars must queue *before* a full street
— at the stop line for a jammed next edge, in the driveway for a jammed
kerbside — rather than be placed on top of the traffic.
"""

from traffic_sim import Car, RandomRouter, TrafficSim, build_grid_network
from traffic_sim.parking import ParkingModel


def _blocker(id, edge_id, s):
    """A car that never moves (max_speed 0): a standing queue member."""
    return Car(id=id, edge_id=edge_id, s=s, v=0.0, max_speed=0.0)


def test_car_waits_at_the_line_when_next_edge_is_full():
    net = build_grid_network(width=3, height=3, block=80.0)
    mover = Car(id=0, edge_id=0, s=0.0, v=0.0)
    target = net.edges[0].v
    next_eid = next(e for e in net.nodes[target].out_edges   # not the U-turn back
                    if net.edges[e].id != 1)
    # Jam the committed next edge right back to the junction.
    blockers = [_blocker(1 + k, next_eid, 1.0 + 7.0 * k) for k in range(3)]
    mover.next_edge = next_eid

    sim = TrafficSim(net, [mover] + blockers, RandomRouter(net, seed=0))
    for _ in range(300):
        sim.step(0.1)

    assert mover.edge_id == 0, "no room past the junction — must wait"
    assert mover.s <= net.edges[0].length
    assert sim.crashes == 0, "waiting at spillback is not a collision"


def test_transfer_slots_in_behind_the_queue_tail():
    net = build_grid_network(width=3, height=3, block=80.0)
    mover = Car(id=0, edge_id=0, s=0.0, v=0.0)
    target = net.edges[0].v
    next_eid = next(e for e in net.nodes[target].out_edges
                    if net.edges[e].id != 1)
    # Queue deep on the next edge but with room near its start to slot into.
    blockers = [_blocker(1 + k, next_eid, 40.0 + 7.0 * k) for k in range(3)]
    mover.next_edge = next_eid

    sim = TrafficSim(net, [mover] + blockers, RandomRouter(net, seed=0))
    for _ in range(400):
        sim.step(0.1)
        if mover.edge_id == next_eid:
            break
    assert mover.edge_id == next_eid, "there was room — the car crosses"
    # It sits behind the tail, never on top of it.
    tail = min(blockers, key=lambda c: c.s)
    assert mover.s <= tail.s - mover.length


def test_parked_car_waits_for_a_kerbside_gap_to_unpark():
    net = build_grid_network(width=3, height=3, block=80.0)
    parked = Car(id=0, edge_id=0, s=20.0, v=0.0, active=False, wake_t=0.0)
    blocker = _blocker(1, 0, 21.0)          # standing right over the spot
    sim = TrafficSim(net, [parked, blocker], RandomRouter(net, seed=0),
                     parking=ParkingModel(seed=0))
    sim.step(0.1)
    assert not parked.active, "blocked in — stays parked"

    blocker.s = 70.0                        # queue clears
    sim.step(0.1)
    assert parked.active, "gap opened — pulls out"
