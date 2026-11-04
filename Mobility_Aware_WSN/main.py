"""
main.py
-------
Runs the complete experiment WITHOUT the dashboard and saves the results.
Use it to check that everything works and to get numbers / images for your report.

Run it with:
    python3 main.py

Output (in the "results" folder):
    speed_comparison.csv        - average results for each speed
    performance_over_time.csv   - step-by-step results of one mobile run
    ml_training_data.csv        - link records used to train the ML model
    speed_comparison.png        - graphs: PDR / loss / delay / connectivity vs speed
    performance_over_time.png   - graph of one mobile run
"""

import os

from performance import summarize_history
from prediction import LinkFailurePredictor, generate_training_data
from simulation import WSNSimulation, run_speed_comparison
from visualization import save_report_charts

RESULTS_FOLDER = "results"
NUM_NODES = 30
COMM_RANGE = 45
AREA_SIZE = 200
STEPS = 100
SPEEDS = [0, 2, 5, 10, 15, 20]


def print_title(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # ---------------------------------------------- 1. single mobile run
    print_title("1. Single simulation run (30 nodes, 5 m/s, 100 steps)")
    sim = WSNSimulation(num_nodes=NUM_NODES, area_size=AREA_SIZE, comm_range=COMM_RANGE,
                        max_speed=5, seed=42)
    history = sim.run(STEPS)
    summary = summarize_history(history)
    print(f"Packet Delivery Ratio : {summary['pdr']:.2f} %")
    print(f"Packet Loss Rate      : {summary['loss_rate']:.2f} %")
    print(f"Average Delay         : {summary['avg_delay_ms']:.2f} ms")
    print(f"Connectivity          : {summary['connectivity']:.2f} %")
    print(f"Active connections    : {summary['active_links']:.1f} links (average)")
    print(f"Reliability Index     : {summary['reliability']:.2f} / 100")
    history.to_csv(os.path.join(RESULTS_FOLDER, "performance_over_time.csv"), index=False)

    # ------------------------------------- 2. stationary vs mobile speeds
    print_title("2. Stationary vs mobile comparison")
    comparison = run_speed_comparison(SPEEDS, num_nodes=NUM_NODES, steps=STEPS,
                                      comm_range=COMM_RANGE, area_size=AREA_SIZE, seed=42)
    print(comparison.round(2).to_string(index=False))
    comparison.to_csv(os.path.join(RESULTS_FOLDER, "speed_comparison.csv"), index=False)

    # ------------------------------------------------ 3. machine learning
    print_title("3. Machine learning: link failure prediction")
    print("Generating training data from simulations at different speeds...")
    data = generate_training_data(comm_range=COMM_RANGE, area_size=AREA_SIZE)
    data.to_csv(os.path.join(RESULTS_FOLDER, "ml_training_data.csv"), index=False)
    print(f"Collected {len(data)} link records "
          f"({data['failed'].mean() * 100:.1f} % of them failed)")

    predictor = LinkFailurePredictor(max_depth=5)
    metrics = predictor.train(data)
    print(f"\nAccuracy  : {metrics['accuracy'] * 100:.2f} %")
    print(f"Precision : {metrics['precision'] * 100:.2f} %")
    print(f"Recall    : {metrics['recall'] * 100:.2f} %")
    print(f"F1-score  : {metrics['f1'] * 100:.2f} %")
    print("\nConfusion matrix [rows = actual, columns = predicted] (0 = stable, 1 = failed):")
    print(metrics["confusion_matrix"])
    print("\nFeature importance:")
    print(metrics["feature_importance"].round(3).to_string())
    print("\nModel comparison:")
    print(metrics["model_comparison"].to_string(index=False))

    # Example predictions
    print("\nExample predictions:")
    examples = [(10, 1, 0), (30, 5, 3), (42, 10, 8), (40, 10, -8)]
    for distance, rel_speed, sep_speed in examples:
        r = predictor.predict_single(distance, rel_speed, sep_speed)
        print(f"  distance={distance:>4} m, relative speed={rel_speed:>3} m/s, "
              f"separation={sep_speed:>3} m/s  ->  {r['predicted_status']:<15}"
              f" risk = {r['failure_risk'] * 100:5.1f} % ({r['risk_level']})")

    # ------------------------------------------------------ 4. save charts
    print_title("4. Saving charts")
    for path in save_report_charts(comparison, history, RESULTS_FOLDER):
        print(f"Saved: {path}")

    print("\nDone! Now start the dashboard with:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
