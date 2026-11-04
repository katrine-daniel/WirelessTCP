"""
simulation.py
-------------
The heart of the project: the WSN simulation engine.

What happens in ONE simulation step (1 step = 1 second):

  1. ROUTE DISCOVERY : every sensor finds its best route to the sink using
                       the CURRENT topology (like a routing table).
  2. MOVEMENT        : all sensors move (Random Direction model).
  3. TRANSMISSION    : every sensor sends 1 data packet to the sink along the
                       route it found in step 1. Each hop:
                         - if the two nodes moved OUT OF RANGE -> link broken
                         - otherwise try up to 4 times; each try succeeds with
                           probability = link quality (distance-based)
  4. MEASUREMENT     : rebuild the topology and calculate PDR, loss, delay,
                       connectivity, active links, reliability.

Why does mobility hurt?  Routes are discovered BEFORE nodes move. The faster
nodes move, the more likely a link in the route has stretched or broken by
the time the packet is sent. A stationary network only suffers from normal
radio errors. This is exactly the real-world problem of mobile WSNs.
"""

import numpy as np
import pandas as pd

from mobility import RandomDirectionMobility
from network import (SINK_ID, build_topology, compute_routes, connected_nodes,
                     distance_matrix, link_features, link_quality)
from performance import compute_step_metrics, history_to_dataframe, summarize_history

# Delay model (milliseconds) for one transmission attempt over one hop
TX_DELAY_MS = 2.0          # time to transmit the packet
PROCESSING_DELAY_MS = 0.5  # time a node needs to process / forward
MAX_BACKOFF_MS = 3.0       # random waiting time to avoid collisions

MAX_DEPLOYMENT_ATTEMPTS = 200  # tries to find a starting layout where all nodes are connected


class WSNSimulation:
    def __init__(self, num_nodes=30, area_size=200, comm_range=45, max_speed=5.0,
                 dt=1.0, max_retries=4, seed=42, record_link_data=True):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.comm_range = comm_range
        self.max_speed = max_speed
        self.dt = dt
        self.max_retries = max_retries
        self.seed = seed
        self.record_link_data = record_link_data
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        """Put the network back to its starting state.

        A real WSN is deployed so that every sensor can reach the sink. We copy
        that: we try several random layouts and keep the first one where all
        nodes are connected (or the best one found if none is fully connected).
        The same seed always gives the same layout, so experiments are fair."""
        centre = self.area_size / 2
        best_mobility = None
        best_connected = -1
        for attempt in range(MAX_DEPLOYMENT_ATTEMPTS):
            mobility = RandomDirectionMobility(
                self.num_nodes, self.area_size, self.max_speed,
                seed=self.seed + attempt * 7919, fixed_nodes=[SINK_ID])
            # Sink / base station sits in the middle of the area and never moves
            mobility.set_position(SINK_ID, centre, centre)
            graph = build_topology(mobility.positions, self.comm_range)
            num_connected = len(connected_nodes(graph, SINK_ID))
            if num_connected > best_connected:
                best_mobility, best_connected = mobility, num_connected
            if num_connected == self.num_nodes:
                break
        self.mobility = best_mobility

        self.channel_rng = np.random.default_rng(self.seed + 1000)
        self.current_step = 0
        self.graph = build_topology(self.mobility.positions, self.comm_range)
        self.history = []          # one dictionary of metrics per step
        self.link_records = []     # data collected for machine learning
        self.last_failed_links = []
        self.last_loss_reasons = {"no_route": 0, "link_broken": 0, "channel_error": 0}

    def set_speed(self, max_speed):
        """Change node speed while the simulation is running."""
        self.max_speed = max_speed
        self.mobility.set_max_speed(max_speed)

    def set_comm_range(self, comm_range):
        """Change the radio range while the simulation is running."""
        self.comm_range = comm_range
        self.graph = build_topology(self.mobility.positions, self.comm_range)

    @property
    def positions(self):
        return self.mobility.positions

    # ------------------------------------------------------------- main step
    def step(self):
        """Advance the simulation by one time step and return its metrics."""
        R = self.comm_range

        # 1. Route discovery on the current topology
        routes = compute_routes(self.graph, SINK_ID)
        old_edges = {tuple(sorted(edge)) for edge in self.graph.edges()}
        pos_before = self.mobility.positions.copy()
        vel_before = self.mobility.velocities.copy()

        # 2. Movement
        self.mobility.update(self.dt)
        pos_after = self.mobility.positions
        dist_after = distance_matrix(pos_after)

        # 3. Packet transmission
        sent = 0
        delivered = 0
        delays = []
        failed_links = []
        loss_reasons = {"no_route": 0, "link_broken": 0, "channel_error": 0}
        links_recorded_this_step = set()

        for source in range(self.num_nodes):
            if source == SINK_ID:
                continue
            sent += 1

            if source not in routes:
                loss_reasons["no_route"] += 1   # node is isolated from the sink
                continue

            path = routes[source]
            packet_delay = 0.0
            packet_ok = True

            for u, v in zip(path[:-1], path[1:]):
                d_now = dist_after[u, v]
                hop_ok = False

                if d_now <= R:
                    q = float(link_quality(d_now, R))
                    for _attempt in range(self.max_retries):
                        packet_delay += (TX_DELAY_MS + PROCESSING_DELAY_MS
                                         + self.channel_rng.uniform(0, MAX_BACKOFF_MS))
                        if self.channel_rng.random() < q:
                            hop_ok = True
                            break
                    reason = "channel_error"
                else:
                    reason = "link_broken"

                # Save data for ML: features are measured BEFORE the move
                # (what a node knows), label = did the link fail?
                link_key = tuple(sorted((u, v)))
                if self.record_link_data and link_key not in links_recorded_this_step:
                    links_recorded_this_step.add(link_key)
                    features = link_features(pos_before[u], pos_before[v],
                                             vel_before[u], vel_before[v])
                    features["speed_setting"] = self.max_speed
                    features["failed"] = 0 if hop_ok else 1
                    self.link_records.append(features)

                if not hop_ok:
                    loss_reasons[reason] += 1
                    failed_links.append((u, v))
                    packet_ok = False
                    break

            if packet_ok:
                delivered += 1
                delays.append(packet_delay)

        # 4. Measurement on the new topology
        self.graph = build_topology(pos_after, R)
        new_edges = {tuple(sorted(edge)) for edge in self.graph.edges()}
        link_breaks = len(old_edges - new_edges)
        num_connected = len(connected_nodes(self.graph, SINK_ID)) - 1  # minus the sink

        metrics = compute_step_metrics(
            sent, delivered, delays, num_connected, self.num_nodes - 1,
            self.graph.number_of_edges(), link_breaks)

        self.current_step += 1
        metrics["step"] = self.current_step
        metrics["speed"] = self.max_speed
        self.history.append(metrics)
        self.last_failed_links = failed_links
        self.last_loss_reasons = loss_reasons
        return metrics

    def run(self, steps):
        """Run several steps without any user interface."""
        for _ in range(steps):
            self.step()
        return self.history_dataframe()

    # --------------------------------------------------------------- helpers
    def history_dataframe(self):
        return history_to_dataframe(self.history)

    def link_dataset(self):
        """All collected link records as a table (for ML training)."""
        return pd.DataFrame(self.link_records)

    def current_link_features(self):
        """Features of every link that exists RIGHT NOW (for live prediction)."""
        rows = []
        pos = self.mobility.positions
        vel = self.mobility.velocities
        for u, v in self.graph.edges():
            features = link_features(pos[u], pos[v], vel[u], vel[v])
            features["node_a"] = u
            features["node_b"] = v
            rows.append(features)
        return pd.DataFrame(rows)

    def node_status(self):
        """Table describing every node: position, speed, connected?, hops."""
        routes = compute_routes(self.graph, SINK_ID)
        pos = self.mobility.positions
        speeds = np.linalg.norm(self.mobility.velocities, axis=1)
        rows = []
        for i in range(self.num_nodes):
            connected = i in routes
            rows.append({
                "node": i,
                "role": "Sink" if i == SINK_ID else "Sensor",
                "x": round(float(pos[i][0]), 1),
                "y": round(float(pos[i][1]), 1),
                "speed_mps": round(float(speeds[i]), 2),
                "neighbours": self.graph.degree(i),
                "connected": connected,
                "hops_to_sink": len(routes[i]) - 1 if connected else None,
            })
        return pd.DataFrame(rows)


def run_speed_comparison(speeds, num_nodes=30, steps=100, comm_range=45,
                         area_size=200, seed=42, repeats=3):
    """For every speed, run the simulation 'repeats' times with different seeds
    and average the results. Every speed uses the SAME set of seeds (same
    starting layouts), so the comparison is fair. Averaging several runs
    removes the effect of one lucky or unlucky layout."""
    rows = []
    for speed in speeds:
        run_summaries = []
        for r in range(repeats):
            sim = WSNSimulation(num_nodes=num_nodes, area_size=area_size,
                                comm_range=comm_range, max_speed=speed,
                                seed=seed + r, record_link_data=False)
            history = sim.run(steps)
            run_summaries.append(summarize_history(history))
        summary = pd.DataFrame(run_summaries).mean().to_dict()
        summary["speed"] = speed
        summary["network_type"] = "Stationary" if speed == 0 else "Mobile"
        rows.append(summary)

    columns = ["speed", "network_type", "pdr", "loss_rate", "avg_delay_ms",
               "connectivity", "active_links", "link_breaks", "reliability"]
    return pd.DataFrame(rows)[columns]
