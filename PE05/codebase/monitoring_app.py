from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import datetime
import time
import logging
import random
import csv

app = Flask(__name__)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Application started: Monitoring and Logging in MLOps")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "logs/monitored_model.pkl")
logging.info("Model trained and saved successfully.")

for i in range(3):
    latency = round(random.uniform(1, 10), 2)
    logging.info(
        f"Simulated log entry {i+1}: Input={[random.random() for _ in range(4)]} "
        f"Output={random.choice([0, 1, 2])} Latency={latency}ms"
    )
CSV_PATH = "logs/request_log.csv"

def append_request_csv(
    timestamp: str,
    features,
    prediction,
    latency_ms,
    is_error: bool = False,
    error_msg: str = "",
) -> None:
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "features", "prediction", "latency_ms", "is_error", "error"])
        writer.writerow([timestamp, features, prediction, latency_ms, is_error, error_msg])

@app.route("/", methods=["GET"])
def home():
    """
    Minimal home route to confirm service availability.
    """
    logging.info("Home route accessed.")
    return "MLOps Monitoring App is running."

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle model predictions and log request details.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = request.get_json(silent=True)

    # Missing JSON or missing 'features'
    if not data or "features" not in data:
        logging.error("Invalid input data: missing JSON body or 'features' key.")
        append_request_csv(timestamp, None, None, None, is_error=True, error_msg="Invalid input data.")
        return jsonify({"error": "Invalid input data."}), 400

    features = data.get("features")
    if not isinstance(features, list) or len(features) != 4:
        logging.error(f"Invalid input data: 'features' must be a list of length 4. Got: {features}")
        append_request_csv(timestamp, features, None, None, is_error=True, error_msg="Invalid input data.")
        return jsonify({"error": "Invalid input data."}), 400

    try:
        features = [float(x) for x in features]
    except (TypeError, ValueError):
        logging.error(f"Invalid input data: 'features' must contain numeric values. Got: {features}")
        append_request_csv(timestamp, features, None, None, is_error=True, error_msg="Invalid input data.")
        return jsonify({"error": "Invalid input data."}), 400

    start_time = time.time()
    prediction = int(model.predict([features])[0])
    latency = round((time.time() - start_time) * 1000, 2)
    correct = None
    if "label" in data:
        try:
            correct = bool(int(data["label"]) == prediction)
        except Exception:
            correct = None

    logging.info(
        f"Prediction made | Input={features} | Output={prediction} | "
        f"Correct={correct} | Latency={latency}ms"
    )
    append_request_csv(timestamp, features, prediction, latency, is_error=False)

    return jsonify(
        {
            "prediction": prediction,
            "timestamp": timestamp,
            "latency_ms": latency,
            "correct": correct,  # None unless label provided
        }
    )

@app.route("/monitor", methods=["GET"])
def monitor():
    """
    Show summary metrics (from app.log).
    """
    try:
        with open("logs/app.log", "r") as f:
            lines = f.readlines()
        recent_logs = lines[-10:]
        total_predictions = sum(1 for line in lines if "Prediction made" in line)
        total_errors = sum(1 for line in lines if "Invalid input data" in line or "ERROR" in line)
    except FileNotFoundError:
        recent_logs = []
        total_predictions = 0
        total_errors = 0

    logging.info("Monitor endpoint accessed.")
    return jsonify(
        {
            "total_predictions": total_predictions,
            "total_errors": total_errors,
            "recent_activity": recent_logs,
        }
    )

@app.route("/health", methods=["GET"])
def health():
    """
    Return a simple status report based on recent activity.
    """
    try:
        with open("logs/app.log", "r") as f:
            lines = f.readlines()

        # Simple heuristic
        last_line = lines[-1] if lines else ""
        healthy = ("ERROR" not in last_line)

    except Exception:
        healthy = False

    status = "healthy" if healthy else "degraded"
    logging.info(f"Health endpoint accessed: Status={status}")
    return jsonify({"status": status, "checked_at": datetime.datetime.now().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)