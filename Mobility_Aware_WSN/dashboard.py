"""
dashboard.py
------------
Interactive Streamlit dashboard.

Run it with:
    streamlit run dashboard.py

How the live animation works (simple version):
  Streamlit runs this whole file from top to bottom every time something
  changes. While the simulation is "running", we do ONE simulation step,
  draw everything, wait a little, and then ask Streamlit to run the file
  again (st.rerun). This repeats until the user presses Pause.
"""

import time

import pandas as pd
import streamlit as st

import visualization as viz
from performance import reliability_label, summarize_history
from prediction import LinkFailurePredictor, generate_training_data
from simulation import WSNSimulation, run_speed_comparison

AREA_SIZE = 200          # metres (square area)
CHART_WINDOW = 300       # how many recent steps to show in live charts

st.set_page_config(page_title="Mobility-Aware WSN Reliability Predictor",
                   page_icon="📡", layout="wide")

# ------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.hero {
    background: linear-gradient(120deg, #0e7490 0%, #1e3a8a 55%, #4c1d95 100%);
    padding: 1.3rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p  {color: #e0f2fe; margin: 0.3rem 0 0 0; font-size: 1.02rem;}
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px; padding: 0.7rem 0.9rem;
}
.status-pill {display:inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
              font-weight: 600; font-size: 0.9rem;}
.running {background:#065f46; color:#d1fae5;}
.paused  {background:#78350f; color:#fef3c7;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📡 Mobility-Aware WSN Reliability Predictor</h1>
  <p>Simulate moving wireless sensor nodes, watch links break, measure network performance
     and predict link failures with machine learning.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------- session state
def create_simulation(num_nodes, speed, comm_range, seed):
    return WSNSimulation(num_nodes=num_nodes, area_size=AREA_SIZE, comm_range=comm_range,
                         max_speed=speed, seed=seed, record_link_data=False)


if "running" not in st.session_state:
    st.session_state.running = False
if "comparison" not in st.session_state:
    st.session_state.comparison = None


@st.cache_resource(show_spinner="Training the ML model on simulated link data (first time only)...")
def get_trained_predictor(comm_range):
    data = generate_training_data(comm_range=comm_range, area_size=AREA_SIZE)
    predictor = LinkFailurePredictor(max_depth=5)
    predictor.train(data)
    return predictor, data


# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    num_nodes = st.slider("Number of sensor nodes", 20, 50, 30, step=1,
                          help="Changing this restarts the simulation.")
    speed = st.slider("Node speed (m/s)", 0.0, 20.0, 5.0, step=0.5,
                      help="0 = stationary network. ~1.5 m/s walking, ~10 m/s car in a city.")
    comm_range = st.slider("Communication range (m)", 30, 70, 45, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1,
                           help="Same seed = same starting positions (repeatable experiment).")
    frame_delay = st.slider("Animation delay (seconds per step)", 0.1, 2.0, 0.4, step=0.1)

    # Create (or re-create) the simulation when nodes/seed change
    sim = st.session_state.get("sim")
    if sim is None or sim.num_nodes != num_nodes or sim.seed != seed:
        st.session_state.sim = create_simulation(num_nodes, speed, comm_range, int(seed))
        st.session_state.running = False
        sim = st.session_state.sim

    # Speed and range can change live
    if sim.max_speed != speed:
        sim.set_speed(speed)
    if sim.comm_range != comm_range:
        sim.set_comm_range(comm_range)

    st.subheader("▶️ Run")
    col1, col2 = st.columns(2)
    if col1.button("Start", type="primary"):
        st.session_state.running = True
    if col2.button("Pause"):
        st.session_state.running = False
    col3, col4 = st.columns(2)
    if col3.button("Reset"):
        sim.reset()
        st.session_state.running = False
    if col4.button("Step +1"):
        st.session_state.running = False
        sim.step()

    state_class = "running" if st.session_state.running else "paused"
    state_text = "● RUNNING" if st.session_state.running else "❚❚ PAUSED"
    st.markdown(f'<span class="status-pill {state_class}">{state_text}</span>',
                unsafe_allow_html=True)
    st.caption(f"Area: {AREA_SIZE} m x {AREA_SIZE} m  |  1 step = 1 second  |  "
               f"Sink = node 0 (centre)")

# ------------------------------------------------ advance the simulation
if st.session_state.running:
    sim.step()

history = sim.history_dataframe()
recent = history.tail(CHART_WINDOW)
latest = history.iloc[-1] if len(history) > 0 else None
previous = history.iloc[-2] if len(history) > 1 else None

predictor, training_data = get_trained_predictor(comm_range)


def delta(column, fmt="{:+.1f}"):
    if latest is None or previous is None:
        return None
    return fmt.format(latest[column] - previous[column])


# --------------------------------------------------------------- tabs
tab_live, tab_graphs, tab_compare, tab_ml, tab_about = st.tabs(
    ["🛰️ Live Network", "📈 Performance Graphs", "⚖️ Mobility Comparison",
     "🤖 ML Failure Prediction", "📘 About the Project"])

# ===== TAB 1: live network ==========================================
with tab_live:
    m = st.columns(6)
    if latest is not None:
        m[0].metric("Packet Delivery Ratio", f"{latest['pdr']:.1f} %", delta("pdr"))
        m[1].metric("Packet Loss Rate", f"{latest['loss_rate']:.1f} %", delta("loss_rate"),
                    delta_color="inverse")
        m[2].metric("Avg End-to-End Delay", f"{latest['avg_delay_ms']:.1f} ms",
                    delta("avg_delay_ms"), delta_color="inverse")
        m[3].metric("Connectivity", f"{latest['connectivity']:.1f} %", delta("connectivity"))
        m[4].metric("Active Connections", int(latest["active_links"]),
                    delta("active_links", "{:+.0f}"))
        m[5].metric("Reliability Index", f"{latest['reliability']:.1f}", delta("reliability"))
    else:
        st.info("Press **Start** (or **Step +1**) in the sidebar to begin the simulation.")

    left, right = st.columns([2.2, 1])
    with left:
        color_mode = st.radio("Colour links by", ["Signal quality", "ML failure risk"],
                              horizontal=True)
        link_predictions = predictor.predict_links(sim.current_link_features())
        fig = viz.topology_figure(
            sim, link_predictions,
            color_mode="risk" if color_mode == "ML failure risk" else "quality")
        st.plotly_chart(fig, key="topology")

    with right:
        score = latest["reliability"] if latest is not None else 0.0
        st.plotly_chart(viz.gauge_figure(score, f"Network Reliability: {reliability_label(score)}"),
                        key="gauge")

        status = sim.node_status()
        sensors = status[status["role"] == "Sensor"]
        n_conn = int(sensors["connected"].sum())
        c1, c2 = st.columns(2)
        c1.metric("🟢 Connected nodes", n_conn)
        c2.metric("🔴 Disconnected nodes", len(sensors) - n_conn)

        st.markdown("**Communication failures (last step)**")
        reasons = sim.last_loss_reasons
        st.dataframe(pd.DataFrame({
            "Failure type": ["Link broken (moved out of range)", "Radio / channel error",
                             "No route to sink (isolated)"],
            "Lost packets": [reasons["link_broken"], reasons["channel_error"],
                             reasons["no_route"]],
        }), hide_index=True)

        high_risk = link_predictions[link_predictions["risk_level"] == "High"] \
            if len(link_predictions) > 0 else link_predictions
        st.markdown(f"**⚠️ High-risk links predicted by ML:** {len(high_risk)}")

    with st.expander("📋 Node status table"):
        st.dataframe(status, hide_index=True)

# ===== TAB 2: performance graphs ====================================
with tab_graphs:
    if len(history) == 0:
        st.info("No data yet. Start the simulation to see live graphs.")
    else:
        summary = summarize_history(history)
        s = st.columns(4)
        s[0].metric("Overall PDR", f"{summary['pdr']:.1f} %")
        s[1].metric("Overall packet loss", f"{summary['loss_rate']:.1f} %")
        s[2].metric("Overall avg delay", f"{summary['avg_delay_ms']:.1f} ms")
        s[3].metric("Average connectivity", f"{summary['connectivity']:.1f} %")

        g1, g2 = st.columns(2)
        g1.plotly_chart(viz.line_chart(recent, "pdr", "Packet Delivery Ratio", viz.COLOR_CONNECTED,
                                       "PDR (%)", [0, 105]), key="g_pdr")
        g2.plotly_chart(viz.line_chart(recent, "loss_rate", "Packet Loss Rate", viz.COLOR_DISCONNECTED,
                                       "Loss (%)", [0, 105]), key="g_loss")
        g3, g4 = st.columns(2)
        g3.plotly_chart(viz.line_chart(recent, "connectivity", "Network Connectivity", viz.COLOR_STRONG,
                                       "Connected nodes (%)", [0, 105]), key="g_conn")
        g4.plotly_chart(viz.line_chart(recent, "avg_delay_ms", "Average End-to-End Delay",
                                       "#a78bfa", "Delay (ms)"), key="g_delay")
        g5, g6 = st.columns(2)
        g5.plotly_chart(viz.line_chart(recent, "active_links", "Active Connections", viz.COLOR_MEDIUM,
                                       "Number of links"), key="g_links")
        g6.plotly_chart(viz.line_chart(recent, "reliability", "Reliability Index", viz.COLOR_SINK,
                                       "Score (0-100)", [0, 105]), key="g_rel")

        st.download_button("⬇️ Download simulation data (CSV)", history.to_csv(index=False),
                           file_name="wsn_simulation_history.csv", mime="text/csv")

# ===== TAB 3: mobility comparison ===================================
with tab_compare:
    st.markdown("Run the **same network** (same seed, same number of nodes, same range) at "
                "different speeds and compare the results. Speed **0 m/s** is the stationary network.")
    c1, c2 = st.columns([3, 1])
    speeds = c1.multiselect("Speeds to compare (m/s)",
                            [0, 1, 2, 3, 5, 8, 10, 12, 15, 20], default=[0, 2, 5, 10, 15, 20])
    steps = c2.number_input("Steps per run", min_value=20, max_value=500, value=100, step=10)

    if st.button("🚀 Run comparison experiment", type="primary"):
        if len(speeds) == 0:
            st.warning("Please choose at least one speed.")
        else:
            st.session_state.running = False  # pause the live view during the experiment
            with st.spinner("Running simulations..."):
                st.session_state.comparison = run_speed_comparison(
                    sorted(speeds), num_nodes=num_nodes, steps=int(steps),
                    comm_range=comm_range, area_size=AREA_SIZE, seed=int(seed))

    comp = st.session_state.comparison
    if comp is not None:
        st.plotly_chart(viz.comparison_lines(comp), key="cmp_lines")
        b1, b2 = st.columns(2)
        b1.plotly_chart(viz.comparison_bar(comp, "pdr", "Packet Delivery Ratio vs Speed", "PDR (%)"),
                        key="cmp_pdr")
        b2.plotly_chart(viz.comparison_bar(comp, "loss_rate", "Packet Loss vs Speed", "Loss (%)",
                                           colorscale="Reds"), key="cmp_loss")
        b3, b4 = st.columns(2)
        b3.plotly_chart(viz.comparison_bar(comp, "avg_delay_ms", "Average Delay vs Speed", "Delay (ms)",
                                           colorscale="Purples"), key="cmp_delay")
        b4.plotly_chart(viz.comparison_bar(comp, "link_breaks", "Link Breaks per Second vs Speed",
                                           "Broken links / step", colorscale="Oranges"), key="cmp_breaks")

        if 0 in list(comp["speed"]) and len(comp) > 1:
            st.subheader("Stationary vs Mobile")
            mobile_options = [float(s) for s in comp["speed"] if s != 0]
            mobile_speed = st.select_slider("Mobile network to compare with",
                                            options=mobile_options, value=max(mobile_options))
            stationary_row = comp[comp["speed"] == 0].iloc[0]
            mobile_row = comp[comp["speed"] == mobile_speed].iloc[0]
            st.plotly_chart(viz.stationary_vs_mobile_figure(stationary_row, mobile_row), key="cmp_svm")
            drop = stationary_row["pdr"] - mobile_row["pdr"]
            st.success(f"At **{mobile_speed:g} m/s** the PDR drops by **{drop:.1f} percentage points** "
                       f"compared to the stationary network, because routes break while packets "
                       f"are being forwarded.")

        st.markdown("**Results table**")
        st.dataframe(comp.round(2), hide_index=True)
        st.download_button("⬇️ Download comparison (CSV)", comp.to_csv(index=False),
                           file_name="mobility_comparison.csv", mime="text/csv")

# ===== TAB 4: machine learning ======================================
with tab_ml:
    metrics = predictor.metrics
    st.markdown("A **Decision Tree** learns from previous transmissions whether a link is likely "
                "to fail, using: distance, signal strength (RSSI), relative speed and separation "
                "speed (moving apart or closer).")

    k = st.columns(4)
    k[0].metric("Accuracy", f"{metrics['accuracy'] * 100:.1f} %")
    k[1].metric("Precision", f"{metrics['precision'] * 100:.1f} %",
                help="Of the links predicted to fail, how many really failed.")
    k[2].metric("Recall", f"{metrics['recall'] * 100:.1f} %",
                help="Of the links that really failed, how many the model caught.")
    k[3].metric("F1-score", f"{metrics['f1'] * 100:.1f} %")
    st.caption(f"Training records: {metrics['train_size']}  |  Test records: {metrics['test_size']}  |  "
               f"Failure rate in data: {metrics['failure_rate'] * 100:.1f} %")

    e1, e2 = st.columns(2)
    e1.plotly_chart(viz.confusion_matrix_figure(metrics["confusion_matrix"]), key="ml_cm")
    e2.plotly_chart(viz.feature_importance_figure(metrics["feature_importance"]), key="ml_fi")

    e3, e4 = st.columns(2)
    with e3:
        st.markdown("**Model comparison**")
        st.dataframe(metrics["model_comparison"], hide_index=True)
        st.markdown("**Classification report**")
        st.dataframe(metrics["report"])
    with e4:
        with st.expander("🌳 Decision tree rules (how the model decides)", expanded=True):
            st.code(predictor.tree_rules(), language="text")

    st.subheader("🔮 Live prediction for the current network")
    if len(link_predictions) == 0:
        st.info("There are no links in the network right now.")
    else:
        p1, p2 = st.columns([1, 1.4])
        p1.plotly_chart(viz.risk_histogram(link_predictions), key="ml_hist")
        table = link_predictions.sort_values("failure_risk", ascending=False).head(15).copy()
        table["failure_risk"] = (table["failure_risk"] * 100).round(1).astype(str) + " %"
        table = table[["node_a", "node_b", "distance", "rssi", "relative_speed",
                       "separation_speed", "predicted_status", "failure_risk", "risk_level"]].round(2)
        p2.markdown("**Top 15 riskiest links right now**")
        p2.dataframe(table, hide_index=True)

    st.subheader("🧪 What-if calculator")
    w1, w2, w3 = st.columns(3)
    w_dist = w1.slider("Distance between nodes (m)", 1.0, float(comm_range), float(comm_range) * 0.6)
    w_rel = w2.slider("Relative speed (m/s)", 0.0, 40.0, 5.0)
    w_sep = w3.slider("Separation speed (m/s, + = moving apart)", -40.0, 40.0, 2.0)
    w_sep = max(-w_rel, min(w_rel, w_sep))  # separation speed cannot exceed relative speed
    result = predictor.predict_single(w_dist, w_rel, w_sep)
    risk_pct = result["failure_risk"] * 100
    if result["predicted_status"] == "Likely to Fail":
        st.error(f"Predicted connection status: **LIKELY TO FAIL**  |  Failure risk: **{risk_pct:.1f} %** "
                 f"({result['risk_level']})")
    else:
        st.success(f"Predicted connection status: **STABLE**  |  Failure risk: **{risk_pct:.1f} %** "
                   f"({result['risk_level']})")

# ===== TAB 5: about =================================================
with tab_about:
    st.markdown("""
### What is this project?
A **Wireless Sensor Network (WSN)** is a group of small battery-powered sensors that send their
readings over radio to a **sink** (base station). In many applications the sensors **move**
(on animals, vehicles, people, robots). This project shows how that movement affects reliability.

### How the simulation works
| Step | What happens |
|------|--------------|
| 1. Route discovery | Each node finds the best path to the sink (lowest ETX = 1 / link quality) |
| 2. Movement | Nodes move with the **Random Direction** model (bounce off walls, stay evenly spread) |
| 3. Transmission | Each node sends one packet. A hop fails if the nodes moved out of range or all 4 radio attempts fail |
| 4. Measurement | PDR, loss, delay, connectivity, active links and reliability are calculated |

### Metrics
* **PDR** = delivered packets / sent packets
* **Packet Loss Rate** = 100 % − PDR
* **Average End-to-End Delay** = time for a delivered packet to reach the sink (transmission + processing + random back-off, including retries)
* **Connectivity** = % of sensors that have a path to the sink
* **Active Connections** = number of working wireless links
* **Reliability Index** = 0.6 × PDR + 0.4 × Connectivity

### Machine learning
A Decision Tree (Scikit-learn) is trained on thousands of link records from simulations at
different speeds. It predicts whether a link will fail and gives a failure-risk percentage.
""")

# --------------------------------------------- keep the animation going
if st.session_state.running:
    time.sleep(frame_delay)
    st.rerun()
