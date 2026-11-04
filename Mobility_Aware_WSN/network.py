"""
network.py
----------
Everything about the WIRELESS LINKS between sensor nodes.

Key ideas (simple radio model):
  * Two nodes can talk only if their distance <= communication range.
  * Closer nodes have a stronger signal (RSSI) and better link quality.
  * Link quality = probability that ONE transmission attempt succeeds.
  * The network topology is stored as a NetworkX graph:
        node  = sensor
        edge  = working wireless link
  * Packets are routed to the sink with the ETX metric
    (Expected Transmission Count = 1 / link quality). A route with a
    small total ETX needs fewer transmissions, so it is more reliable.
"""

import numpy as np
import networkx as nx

SINK_ID = 0  # node 0 is the sink / base station


def distance_matrix(positions):
    """Return an N x N matrix where entry [i, j] is the distance between nodes i and j."""
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def rssi_dbm(distance):
    """Received signal strength using the log-distance path-loss model.

    RSSI = Tx power - (PL0 + 10 * n * log10(d / d0))
      Tx power = 0 dBm, PL0 = 40 dB at d0 = 1 m, path-loss exponent n = 2.7
    """
    d = np.maximum(distance, 1.0)  # avoid log(0)
    return 0.0 - (40.0 + 10 * 2.7 * np.log10(d))


def link_quality(distance, comm_range):
    """Probability (0..1) that a single transmission over this link succeeds.

    Very close  -> about 1.0 (almost always succeeds)
    At the edge -> about 0.1 (usually fails)
    Out of range -> 0.0
    """
    distance = np.asarray(distance, dtype=float)
    quality = 1.0 - 0.9 * (distance / comm_range) ** 4
    return np.where(distance <= comm_range, quality, 0.0)


def build_topology(positions, comm_range):
    """Build the network graph: an edge exists when two nodes are within range."""
    graph = nx.Graph()
    num_nodes = len(positions)
    for i in range(num_nodes):
        graph.add_node(i, pos=(float(positions[i][0]), float(positions[i][1])))

    dist = distance_matrix(positions)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = dist[i, j]
            if d <= comm_range:
                q = float(link_quality(d, comm_range))
                graph.add_edge(i, j, distance=float(d), quality=q, etx=1.0 / max(q, 0.01))
    return graph


def compute_routes(graph, sink=SINK_ID):
    """Best (lowest ETX) route from every reachable node to the sink.

    Returns a dictionary: {node_id: [node_id, next_hop, ..., sink]}
    """
    paths_from_sink = nx.single_source_dijkstra_path(graph, sink, weight="etx")
    # Dijkstra gives sink -> node paths; reverse them to get node -> sink
    return {node: list(reversed(path)) for node, path in paths_from_sink.items()}


def connected_nodes(graph, sink=SINK_ID):
    """Set of nodes that have at least one path to the sink."""
    return nx.node_connected_component(graph, sink)


def link_features(pos_u, pos_v, vel_u, vel_v):
    """Features describing one link, used by the machine-learning model.

    distance         : current distance between the two nodes (m)
    rssi             : received signal strength (dBm)
    relative_speed   : how fast the two nodes move relative to each other (m/s)
    separation_speed : > 0 means the nodes are moving APART,
                       < 0 means they are moving CLOSER (m/s)
    """
    rel_pos = np.asarray(pos_v) - np.asarray(pos_u)
    rel_vel = np.asarray(vel_v) - np.asarray(vel_u)
    distance = float(np.linalg.norm(rel_pos))
    relative_speed = float(np.linalg.norm(rel_vel))
    if distance > 0:
        separation_speed = float(np.dot(rel_pos, rel_vel) / distance)
    else:
        separation_speed = 0.0
    return {
        "distance": distance,
        "rssi": float(rssi_dbm(distance)),
        "relative_speed": relative_speed,
        "separation_speed": separation_speed,
    }
