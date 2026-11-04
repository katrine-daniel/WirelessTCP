# 📡 Mobility-Aware WSN Reliability Predictor

A fully software-based simulation of a **Wireless Sensor Network (WSN)** with **moving sensor nodes**.
It shows how node mobility affects connectivity, packet delivery, packet loss and delay, and uses
**machine learning** to predict which wireless links are likely to fail.

> No hardware needed. No Arduino, ESP32 or real sensors. Everything is simulated in Python.

---

## 1. Features

| Area | What it does |
|------|--------------|
| Simulation | 20–50 virtual sensor nodes in a 200 m × 200 m area, one fixed sink (base station) |
| Mobility | Random Direction model (nodes bounce off walls), speed adjustable live from 0 (stationary) to 20 m/s |
| Communication | Distance-based radio model (RSSI, link quality), multi-hop routing, up to 4 attempts per hop (like IEEE 802.15.4) |
| Failures | Detects broken links, radio errors and isolated (disconnected) nodes |
| Metrics | PDR, packet loss rate, average end-to-end delay, connectivity, active connections, reliability index |
| Comparison | Stationary vs mobile, and many speeds side by side |
| Dashboard | Streamlit app: live topology map, Start/Pause/Reset, sliders, live graphs, gauges |
| Machine learning | Decision Tree predicts link failure, shows risk %, accuracy, confusion matrix, rules |

---

## 2. Project structure

```
Mobility_Aware_WSN/
├── main.py            # Runs the full experiment in the terminal and saves results
├── dashboard.py       # Streamlit interactive dashboard
├── simulation.py      # Simulation engine (one step = route, move, transmit, measure)
├── mobility.py        # Random Direction mobility model
├── network.py         # Radio model, topology graph (NetworkX), routing
├── performance.py     # Formulas for PDR, loss, delay, connectivity, reliability
├── prediction.py      # Machine learning (Scikit-learn Decision Tree)
├── visualization.py   # Plotly charts (dashboard) + Matplotlib charts (report images)
├── requirements.txt   # Python libraries
├── README.md          # This file
└── .streamlit/
    └── config.toml    # Dark theme for the dashboard
```

How the files work together:

```
mobility.py ─┐
network.py  ─┼─► simulation.py ─► prediction.py ─┐
performance.py┘         │                        ├─► dashboard.py  (interactive)
                        └──► visualization.py ───┴─► main.py       (terminal + saved files)
```

---

## 3. Setup on a MacBook (step by step)

### Step 1 – Install Python
Open **Terminal** (Cmd + Space, type "Terminal") and check:

```bash
python3 --version
```

You need **Python 3.10 or newer**. If it is missing or older, download it from
<https://www.python.org/downloads/macos/> and install it (choose the "macOS 64-bit universal2 installer",
it works on M1/M2).

### Step 2 – Install VS Code + Python extension
1. Download VS Code from <https://code.visualstudio.com/> and drag it into Applications.
2. Open VS Code → Extensions (Cmd + Shift + X) → search **Python** (by Microsoft) → Install.

### Step 3 – Open the project folder
In VS Code: **File → Open Folder…** → choose `Mobility_Aware_WSN`.
Then open the built-in terminal: **Terminal → New Terminal** (or Ctrl + `).

### Step 4 – Create a virtual environment (a private box for the libraries)

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

You should now see `(venv)` at the start of the terminal line.
In VS Code press Cmd + Shift + P → **Python: Select Interpreter** → choose the one that shows `venv`.

### Step 5 – Install the libraries

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 6 – Run the terminal experiment (checks that everything works)

```bash
python3 main.py
```

It prints the results and creates a `results/` folder with CSV files and PNG graphs for your report.

### Step 7 – Start the dashboard

```bash
streamlit run dashboard.py
```

Your browser opens at <http://localhost:8501>. To stop the dashboard press **Ctrl + C** in the terminal.

> Next time you open the project you only need: `source venv/bin/activate` and then `streamlit run dashboard.py`.

---

## 4. Using the dashboard (demo script)

1. **Sidebar:** choose number of nodes, speed, communication range. Press **Start**.
2. **🛰️ Live Network:** watch nodes move. Cyan = connected, red = disconnected, gold star = sink.
   Green/amber/orange lines = strong/medium/weak links. Red dotted lines = failed transmissions.
   Switch "Colour links by" to **ML failure risk** to see the model's predictions on the map.
3. Move the **speed slider** from 0 to 20 while running – PDR drops and failures appear.
4. **📈 Performance Graphs:** live PDR, loss, connectivity, delay, active links, reliability.
5. **⚖️ Mobility Comparison:** click **Run comparison experiment** to compare 0 m/s (stationary)
   with mobile speeds. Shows the "Stationary vs Mobile" chart.
6. **🤖 ML Failure Prediction:** accuracy, precision, recall, confusion matrix, feature importance,
   decision-tree rules, live risky links, and a **what-if calculator**.

---

## 5. How it works

### 5.1 Simulation step (1 step = 1 second)
1. **Route discovery** – each node finds the best route to the sink using Dijkstra with the
   **ETX** metric (Expected Transmission Count = 1 / link quality).
2. **Movement** – nodes move (Random Direction model). Random Waypoint was avoided on purpose: in that model nodes bunch up in the centre, which would make mobile networks look denser (and better) than the stationary one.
3. **Transmission** – each sensor sends 1 packet along its route. For every hop:
   - nodes now farther apart than the range → **link broken**
   - otherwise up to **4 attempts** (1 send + 3 retransmissions, as in IEEE 802.15.4), each succeeding with probability = link quality
4. **Measurement** – the topology graph is rebuilt and all metrics are calculated.

**Why mobility hurts:** the route was chosen *before* the nodes moved. The faster they move, the more
links in the route stretch or break before the packet is forwarded.

### 5.2 Radio model
* Communication range *R* (default 45 m)
* RSSI (dBm) = 0 − (40 + 27 · log10(d)) (log-distance path loss, exponent 2.7)
* Link quality q = 1 − 0.9 · (d / R)⁴ for d ≤ R, otherwise 0
* Delay per attempt = 2 ms transmission + 0.5 ms processing + random 0–3 ms back-off

### 5.3 Performance metrics

| Metric | Formula |
|--------|---------|
| Packet Delivery Ratio | delivered ÷ sent × 100 |
| Packet Loss Rate | (sent − delivered) ÷ sent × 100 |
| Average End-to-End Delay | mean delay of delivered packets (ms) |
| Network Connectivity | sensors with a path to the sink ÷ all sensors × 100 |
| Active Connections | number of edges in the topology graph |
| Reliability Index | 0.6 × PDR + 0.4 × Connectivity (0–100) |

### 5.4 Machine learning
* **Data:** simulations at 0, 2, 5, 8, 12, 16 and 20 m/s; every used link becomes one record.
* **Features:** distance, RSSI, relative speed, separation speed (+ = moving apart).
* **Label:** `failed` = 1 if the packet could not cross the link.
* **Model:** `DecisionTreeClassifier(max_depth=5, class_weight="balanced")` – it learns readable
  IF-ELSE rules. Logistic Regression is trained too, as a baseline for comparison.
* **Evaluation:** 75 % train / 25 % test, accuracy, precision, recall, F1-score, confusion matrix.
* **Output:** predicted status (Stable / Likely to Fail) and failure risk (%) for each link.

---

## 6. Expected results (typical)

Output of `python3 main.py` (30 nodes, 45 m range, 100 steps, averaged over 3 seeds):

| Speed (m/s) | PDR (%) | Packet loss (%) | Avg delay (ms) | Connectivity (%) | Link breaks / s | Reliability |
|------|------|------|------|-------|------|------|
| 0 (stationary) | 89.3 | 10.7 | 21.5 | 100.0 | 0.0 | 93.6 |
| 2  | 64.9 | 35.1 | 16.6 | 76.6 | 1.4  | 69.6 |
| 5  | 60.6 | 39.4 | 15.7 | 78.1 | 3.6  | 67.6 |
| 10 | 50.3 | 49.7 | 14.2 | 78.6 | 7.2  | 61.6 |
| 15 | 35.4 | 64.6 | 13.2 | 73.7 | 10.3 | 50.7 |
| 20 | 30.3 | 69.7 | 11.9 | 77.5 | 14.3 | 49.2 |

**How to explain these results:**
* **PDR falls steadily with speed** because link breaks per second grow almost linearly with speed.
  Routes chosen at the start of a step are already broken when packets are forwarded.
* **Connectivity drops from 100 % to about 77 %** as soon as nodes move. The stationary network
  was deployed so that all nodes are connected; moving nodes keep drifting out of range of their
  neighbours and become isolated for a while.
* **Delay goes DOWN with speed.** This is a known effect called *survivor bias*: at high
  speed the long, many-hop routes break, so only packets from nodes close to the sink get through,
  and those packets are naturally fast. The lost packets are not counted in the delay.
* **ML:** the Decision Tree reaches about 86 % accuracy with about 86 % recall (it catches most
  failing links). *Separation speed* and *distance* are its most important features, which shows
  that "moving apart near the edge of range" is the main reason links fail.

Exact numbers change if you change the number of nodes, the range or the random seed.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `command not found: python` | On Mac use `python3` and `pip` inside the venv |
| `ModuleNotFoundError: No module named 'streamlit'` | Activate the venv (`source venv/bin/activate`) then `pip install -r requirements.txt` |
| `command not found: streamlit` | Same as above, or run `python3 -m streamlit run dashboard.py` |
| Port 8501 already in use | `streamlit run dashboard.py --server.port 8502` |
| VS Code shows yellow import warnings | Cmd + Shift + P → *Python: Select Interpreter* → pick the `venv` one |
| Dashboard is slow | Increase "Animation delay" or reduce the number of nodes |

---

## 8. Possible future work
* Energy model (battery drain per transmission) and network lifetime
* Other mobility models (Gauss-Markov, group mobility)
* Mobility-aware routing that avoids high-risk links using the ML prediction
* Comparing more ML models (Random Forest, SVM)

## 9. Technologies
Python · NumPy · Pandas · NetworkX · Plotly · Matplotlib · Streamlit · Scikit-learn
