"""
performance.py
--------------
Formulas for the network performance metrics.

  Packet Delivery Ratio (PDR) = packets delivered / packets sent
  Packet Loss Rate            = packets lost / packets sent  (= 1 - PDR)
  Average End-to-End Delay    = average time a delivered packet takes to reach the sink
  Network Connectivity        = % of sensor nodes that have a path to the sink
  Active Connections          = number of working wireless links
  Network Reliability Index   = 0.6 * PDR + 0.4 * Connectivity   (our combined score)
"""

import numpy as np
import pandas as pd


def packet_delivery_ratio(delivered, sent):
    return delivered / sent if sent > 0 else 0.0


def packet_loss_rate(delivered, sent):
    return 1.0 - packet_delivery_ratio(delivered, sent) if sent > 0 else 0.0


def average_delay(delays):
    """Average of the list of end-to-end delays (only delivered packets)."""
    return float(np.mean(delays)) if len(delays) > 0 else 0.0


def network_connectivity(num_connected, num_sensors):
    """Fraction of sensors (not counting the sink) that can reach the sink."""
    return num_connected / num_sensors if num_sensors > 0 else 0.0


def reliability_index(pdr, connectivity):
    """A single 0-100 score. PDR is weighted more because delivering data is
    the main job of a WSN; connectivity shows how 'healthy' the topology is."""
    return 100.0 * (0.6 * pdr + 0.4 * connectivity)


def reliability_label(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def compute_step_metrics(sent, delivered, delays, num_connected, num_sensors,
                         active_links, link_breaks):
    """Calculate all metrics for ONE simulation step and return a dictionary."""
    pdr = packet_delivery_ratio(delivered, sent)
    connectivity = network_connectivity(num_connected, num_sensors)
    return {
        "sent": sent,
        "delivered": delivered,
        "lost": sent - delivered,
        "pdr": pdr * 100,
        "loss_rate": packet_loss_rate(delivered, sent) * 100,
        "avg_delay_ms": average_delay(delays),
        "connectivity": connectivity * 100,
        "active_links": active_links,
        "link_breaks": link_breaks,
        "reliability": reliability_index(pdr, connectivity),
    }


def summarize_history(history_df):
    """Overall results of a whole simulation run.

    PDR and loss are computed from TOTAL packets (more accurate than
    averaging the per-step percentages)."""
    if history_df is None or len(history_df) == 0:
        return {
            "pdr": 0.0, "loss_rate": 0.0, "avg_delay_ms": 0.0, "connectivity": 0.0,
            "active_links": 0.0, "link_breaks": 0.0, "reliability": 0.0,
        }

    total_sent = history_df["sent"].sum()
    total_delivered = history_df["delivered"].sum()
    pdr = packet_delivery_ratio(total_delivered, total_sent)
    connectivity = history_df["connectivity"].mean() / 100

    # Delay: weight each step's delay by how many packets were delivered in it
    if total_delivered > 0:
        delay = float((history_df["avg_delay_ms"] * history_df["delivered"]).sum() / total_delivered)
    else:
        delay = 0.0

    return {
        "pdr": pdr * 100,
        "loss_rate": (1 - pdr) * 100,
        "avg_delay_ms": delay,
        "connectivity": connectivity * 100,
        "active_links": float(history_df["active_links"].mean()),
        "link_breaks": float(history_df["link_breaks"].mean()),
        "reliability": reliability_index(pdr, connectivity),
    }


def history_to_dataframe(history):
    """Convert the list of per-step dictionaries into a Pandas table."""
    return pd.DataFrame(history)
