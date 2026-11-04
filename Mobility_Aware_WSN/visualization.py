"""
visualization.py
----------------
All charts of the project.

  * Plotly figures  -> interactive charts used in the Streamlit dashboard
  * Matplotlib      -> PNG images saved by main.py (for your project report)
"""

import os

import matplotlib
matplotlib.use("Agg")  # draw to files only, no pop-up windows
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from network import SINK_ID

# ---------------------------------------------------------------- colours
COLOR_CONNECTED = "#22d3ee"     # cyan   - sensor that can reach the sink
COLOR_DISCONNECTED = "#f87171"  # red    - isolated sensor
COLOR_SINK = "#facc15"          # gold   - sink / base station
COLOR_STRONG = "#34d399"        # green  - strong link
COLOR_MEDIUM = "#fbbf24"        # amber  - medium link
COLOR_WEAK = "#fb923c"          # orange - weak link
COLOR_HIGH_RISK = "#ef4444"     # red    - high failure risk
COLOR_FAILED = "#ef4444"

BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _edge_trace(segments, color, width, name, dash="solid"):
    """Draw many line segments as ONE trace (much faster than one per edge)."""
    xs, ys = [], []
    for (x0, y0), (x1, y1) in segments:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name, hoverinfo="skip",
                      line=dict(color=color, width=width, dash=dash))


def topology_figure(sim, link_predictions=None, color_mode="quality", show_range=True):
    """Network map: nodes, links, failures.

    color_mode = "quality" -> links coloured by signal quality
    color_mode = "risk"    -> links coloured by ML failure risk
    """
    pos = sim.positions
    fig = go.Figure()

    # ----- links
    groups = {}
    if color_mode == "risk" and link_predictions is not None and len(link_predictions) > 0:
        style = {"Low": (COLOR_STRONG, 2, "Low risk link"),
                 "Medium": (COLOR_MEDIUM, 2.5, "Medium risk link"),
                 "High": (COLOR_HIGH_RISK, 3, "High risk link")}
        for _, row in link_predictions.iterrows():
            a, b = int(row["node_a"]), int(row["node_b"])
            groups.setdefault(row["risk_level"], []).append((pos[a], pos[b]))
    else:
        style = {"Strong": (COLOR_STRONG, 2, "Strong link"),
                 "Medium": (COLOR_MEDIUM, 1.8, "Medium link"),
                 "Weak": (COLOR_WEAK, 1.2, "Weak link")}
        for a, b, data in sim.graph.edges(data=True):
            q = data["quality"]
            level = "Strong" if q >= 0.7 else ("Medium" if q >= 0.4 else "Weak")
            groups.setdefault(level, []).append((pos[a], pos[b]))

    for level, (color, width, name) in style.items():
        if level in groups:
            fig.add_trace(_edge_trace(groups[level], color, width, name))

    # ----- links that failed in the last step (dashed red)
    if sim.last_failed_links:
        failed_segments = [(pos[a], pos[b]) for a, b in sim.last_failed_links]
        fig.add_trace(_edge_trace(failed_segments, COLOR_FAILED, 2, "Failed transmission", "dot"))

    # ----- nodes
    status = sim.node_status()
    sensors = status[status["role"] == "Sensor"]
    for is_connected, color, name in [(True, COLOR_CONNECTED, "Connected node"),
                                      (False, COLOR_DISCONNECTED, "Disconnected node")]:
        subset = sensors[sensors["connected"] == is_connected]
        if len(subset) == 0:
            continue
        hover = [
            f"<b>Node {r.node}</b><br>Position: ({r.x}, {r.y})<br>Speed: {r.speed_mps} m/s"
            f"<br>Neighbours: {r.neighbours}<br>Hops to sink: "
            f"{int(r.hops_to_sink) if r.connected else 'No route'}"
            for r in subset.itertuples()
        ]
        fig.add_trace(go.Scatter(
            x=subset["x"], y=subset["y"], mode="markers+text", name=name,
            text=subset["node"].astype(str), textposition="top center",
            textfont=dict(size=9, color="#cbd5e1"),
            marker=dict(size=13, color=color, line=dict(width=1.5, color="#0f172a")),
            hovertext=hover, hoverinfo="text"))

    sink = status[status["role"] == "Sink"].iloc[0]
    fig.add_trace(go.Scatter(
        x=[sink["x"]], y=[sink["y"]], mode="markers", name="Sink (base station)",
        marker=dict(size=24, color=COLOR_SINK, symbol="star", line=dict(width=1.5, color="#0f172a")),
        hovertext=f"<b>Sink node {SINK_ID}</b><br>Collects all data", hoverinfo="text"))

    # ----- dashed circle showing the sink's radio range
    shapes = []
    if show_range:
        r = sim.comm_range
        shapes.append(dict(type="circle", x0=sink["x"] - r, y0=sink["y"] - r,
                           x1=sink["x"] + r, y1=sink["y"] + r,
                           line=dict(color=COLOR_SINK, width=1, dash="dash"), opacity=0.4))

    fig.update_layout(
        **BASE_LAYOUT, height=620, shapes=shapes,
        title=f"Network Topology  -  step {sim.current_step}",
        legend=dict(orientation="h", y=-0.08, x=0),
        xaxis=dict(range=[-5, sim.area_size + 5], title="X position (m)",
                   showgrid=True, gridcolor="rgba(148,163,184,0.1)", zeroline=False),
        yaxis=dict(range=[-5, sim.area_size + 5], title="Y position (m)",
                   showgrid=True, gridcolor="rgba(148,163,184,0.1)", zeroline=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def line_chart(history_df, column, title, color, y_title, y_range=None):
    """Live time-series chart (e.g. PDR over time)."""
    fig = go.Figure()
    if len(history_df) > 0:
        fig.add_trace(go.Scatter(
            x=history_df["step"], y=history_df[column], mode="lines", name=title,
            line=dict(color=color, width=2.5), fill="tozeroy",
            fillcolor=_transparent(color, 0.15)))
    fig.update_layout(**BASE_LAYOUT, height=300, title=title, showlegend=False,
                      xaxis_title="Simulation step (s)", yaxis_title=y_title)
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig


def gauge_figure(value, title):
    """Speedometer-style gauge for the reliability index (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=" / 100"),
        title=dict(text=title, font=dict(size=16)),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="#e2e8f0"),
            steps=[dict(range=[0, 50], color="#7f1d1d"),
                   dict(range=[50, 70], color="#92400e"),
                   dict(range=[70, 85], color="#365314"),
                   dict(range=[85, 100], color="#065f46")]),
    ))
    fig.update_layout(**{**BASE_LAYOUT, "margin": dict(l=20, r=20, t=60, b=10)}, height=260)
    return fig


def comparison_bar(comp_df, column, title, y_title, colorscale="Tealgrn"):
    """Bar chart: one bar per speed."""
    labels = [f"{s:g} m/s" for s in comp_df["speed"]]
    fig = go.Figure(go.Bar(
        x=labels, y=comp_df[column], text=comp_df[column].round(1), textposition="outside",
        marker=dict(color=comp_df[column], colorscale=colorscale)))
    fig.update_layout(**BASE_LAYOUT, height=340, title=title,
                      xaxis_title="Node speed", yaxis_title=y_title)
    return fig


def comparison_lines(comp_df):
    """PDR, connectivity and reliability vs speed on one chart."""
    fig = go.Figure()
    for column, name, color in [("pdr", "PDR (%)", COLOR_CONNECTED),
                                ("connectivity", "Connectivity (%)", COLOR_STRONG),
                                ("reliability", "Reliability index", COLOR_SINK),
                                ("loss_rate", "Packet loss (%)", COLOR_DISCONNECTED)]:
        fig.add_trace(go.Scatter(x=comp_df["speed"], y=comp_df[column], name=name,
                                 mode="lines+markers", line=dict(color=color, width=3),
                                 marker=dict(size=9)))
    fig.update_layout(**BASE_LAYOUT, height=420, title="Effect of Node Speed on Network Performance",
                      xaxis_title="Node speed (m/s)", yaxis_title="Percentage / score",
                      yaxis=dict(range=[0, 105]), legend=dict(orientation="h", y=-0.2))
    return fig


def stationary_vs_mobile_figure(stationary_row, mobile_row):
    """Grouped bar chart: stationary network vs a mobile network."""
    metrics = ["pdr", "loss_rate", "connectivity", "reliability"]
    names = ["PDR (%)", "Packet loss (%)", "Connectivity (%)", "Reliability"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Stationary (0 m/s)", x=names,
                         y=[stationary_row[m] for m in metrics], marker_color=COLOR_STRONG,
                         text=[f"{stationary_row[m]:.1f}" for m in metrics], textposition="outside"))
    fig.add_trace(go.Bar(name=f"Mobile ({mobile_row['speed']:g} m/s)", x=names,
                         y=[mobile_row[m] for m in metrics], marker_color=COLOR_DISCONNECTED,
                         text=[f"{mobile_row[m]:.1f}" for m in metrics], textposition="outside"))
    fig.update_layout(**BASE_LAYOUT, height=400, barmode="group",
                      title="Stationary vs Mobile Sensor Network",
                      yaxis=dict(range=[0, 115]), legend=dict(orientation="h", y=-0.15))
    return fig


def confusion_matrix_figure(cm):
    labels = ["Stable", "Failed"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
        text=cm, texttemplate="%{text}", textfont=dict(size=20),
        colorscale="Blues", showscale=False))
    fig.update_layout(**BASE_LAYOUT, height=360, title="Confusion Matrix (test data)")
    fig.update_yaxes(autorange="reversed")
    return fig


def feature_importance_figure(importance_series):
    s = importance_series.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=COLOR_CONNECTED,
                           text=[f"{v:.2f}" for v in s.values], textposition="outside"))
    fig.update_layout(**BASE_LAYOUT, height=360, title="Which features matter most?",
                      xaxis_title="Importance", xaxis=dict(range=[0, max(1.0, s.max() * 1.2)]))
    return fig


def risk_histogram(predictions_df):
    fig = go.Figure(go.Histogram(x=predictions_df["failure_risk"] * 100, nbinsx=20,
                                 marker_color=COLOR_MEDIUM))
    fig.update_layout(**BASE_LAYOUT, height=300, title="Failure risk of current links",
                      xaxis_title="Predicted failure risk (%)", yaxis_title="Number of links")
    return fig


def _transparent(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------- Matplotlib (saved images)
def save_report_charts(comp_df, history_df, folder="results"):
    """Save PNG charts that you can paste into your project report."""
    os.makedirs(folder, exist_ok=True)
    saved = []

    # 1. Effect of speed: 2 x 2 grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plots = [("pdr", "Packet Delivery Ratio (%)", "tab:blue"),
             ("loss_rate", "Packet Loss Rate (%)", "tab:red"),
             ("avg_delay_ms", "Average End-to-End Delay (ms)", "tab:purple"),
             ("connectivity", "Network Connectivity (%)", "tab:green")]
    for ax, (column, title, color) in zip(axes.flat, plots):
        ax.plot(comp_df["speed"], comp_df[column], marker="o", color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Node speed (m/s)")
        ax.grid(alpha=0.3)
    fig.suptitle("Effect of Node Mobility on WSN Performance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(folder, "speed_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 2. Time series of one mobile run
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history_df["step"], history_df["pdr"], label="PDR (%)", color="tab:blue")
    ax.plot(history_df["step"], history_df["connectivity"], label="Connectivity (%)", color="tab:green")
    ax.plot(history_df["step"], history_df["loss_rate"], label="Packet loss (%)", color="tab:red")
    ax.set_xlabel("Simulation step (s)")
    ax.set_ylabel("Percentage")
    ax.set_title("Mobile Network Performance Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(folder, "performance_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    return saved
