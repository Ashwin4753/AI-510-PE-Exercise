import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = os.path.join("model", "iris_model.pkl")

app = Flask(__name__)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)

model = None

@app.before_request
def ensure_model_loaded():
    global model
    if model is None:
        model = load_model()

@app.get("/")
def home():
    return jsonify({"status": "ok", "message": "Iris model API running"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Invalid input. Provide JSON: {\"features\": [f1, f2, f3, f4]}"
        }), 400

    try:
        X = np.array([features], dtype=float)
    except Exception:
        return jsonify({"error": "Features must be numeric."}), 400

    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist()

    return jsonify({
        "prediction": pred,
        "probabilities": probs
    })

if __name__ == "__main__":
    # For local dev in Codespaces
    app.run(host="0.0.0.0", port=5000)