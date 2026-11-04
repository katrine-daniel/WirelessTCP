"""
prediction.py
-------------
Machine learning: predict whether a wireless link is likely to FAIL.

1. DATA      : we run the simulation at different speeds and record every
               link that was used to forward a packet.
               Features (what a node can measure):
                 - distance          (m)
                 - rssi              (signal strength, dBm)
                 - relative_speed    (m/s)
                 - separation_speed  (m/s, > 0 = moving apart)
               Label: failed = 1 if the packet could not cross the link.

2. MODEL     : a Decision Tree classifier. It is easy to explain because it
               learns simple IF-ELSE rules, for example:
                 IF distance > 38 m AND separation_speed > 2 m/s -> FAIL

3. EVALUATION: 75% of the data trains the model, 25% tests it.
               We report accuracy, precision, recall, F1-score, a confusion
               matrix, and compare with Logistic Regression as a baseline.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from network import rssi_dbm
from simulation import WSNSimulation

FEATURES = ["distance", "rssi", "relative_speed", "separation_speed"]
LABEL = "failed"


def generate_training_data(speeds=(0, 2, 5, 8, 12, 16, 20), num_nodes=30, steps=60,
                           comm_range=45, area_size=200, seed=7):
    """Run simulations at several speeds and collect link records."""
    all_data = []
    for i, speed in enumerate(speeds):
        sim = WSNSimulation(num_nodes=num_nodes, area_size=area_size,
                            comm_range=comm_range, max_speed=speed,
                            seed=seed + i, record_link_data=True)
        sim.run(steps)
        all_data.append(sim.link_dataset())
    return pd.concat(all_data, ignore_index=True)


def risk_level(probability):
    if probability >= 0.6:
        return "High"
    if probability >= 0.3:
        return "Medium"
    return "Low"


class LinkFailurePredictor:
    def __init__(self, max_depth=5):
        # class_weight="balanced": failures are rarer than successes, so we tell
        # the model to treat both classes as equally important.
        self.model = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                            min_samples_leaf=10, random_state=42)
        self.baseline = make_pipeline(StandardScaler(),
                                      LogisticRegression(class_weight="balanced", max_iter=1000))
        self.metrics = None
        self.is_trained = False

    def train(self, data):
        """Train on a DataFrame with the FEATURES columns and a 'failed' column."""
        if data[LABEL].nunique() < 2:
            raise ValueError("Training data needs both failed and successful links. "
                             "Run more steps or use higher speeds.")

        X = data[FEATURES]
        y = data[LABEL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        self.model.fit(X_train, y_train)
        self.baseline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        y_pred_baseline = self.baseline.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=["Stable (0)", "Failed (1)"],
                                       output_dict=True, zero_division=0)

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
            "report": pd.DataFrame(report).transpose().round(3),
            "feature_importance": pd.Series(self.model.feature_importances_, index=FEATURES)
                                    .sort_values(ascending=False),
            "model_comparison": pd.DataFrame({
                "Model": ["Decision Tree", "Logistic Regression"],
                "Accuracy": [accuracy_score(y_test, y_pred),
                             accuracy_score(y_test, y_pred_baseline)],
                "F1-score": [f1_score(y_test, y_pred, zero_division=0),
                             f1_score(y_test, y_pred_baseline, zero_division=0)],
            }).round(3),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "failure_rate": float(y.mean()),
        }
        return self.metrics

    def tree_rules(self):
        """The decision tree written as readable IF-ELSE rules."""
        return export_text(self.model, feature_names=FEATURES, decimals=1)

    def predict_links(self, link_df):
        """Add predicted status and failure risk to a table of links."""
        result = link_df.copy()
        if len(result) == 0:
            for col in ["failure_risk", "risk_level", "predicted_status"]:
                result[col] = []
            return result
        probabilities = self.model.predict_proba(result[FEATURES])[:, 1]
        result["failure_risk"] = probabilities
        result["risk_level"] = [risk_level(p) for p in probabilities]
        result["predicted_status"] = ["Likely to Fail" if p >= 0.5 else "Stable"
                                      for p in probabilities]
        return result

    def predict_single(self, distance, relative_speed, separation_speed):
        """Predict for one link entered by the user (what-if calculator)."""
        row =pd.DataFrame([{
            "distance": distance,
            "rssi": float(rssi_dbm(distance)),
            "relative_speed": relative_speed,
            "separation_speed": separation_speed,
        }])
        return self.predict_links(row).iloc[0]
