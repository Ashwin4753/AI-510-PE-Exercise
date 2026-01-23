from flask import Flask, request, jsonify
import joblib
import numpy as np
import sys
import platform
import socket

app = Flask(__name__)

# Load trained model
model = joblib.load("/workspaces/AI-510-PE-Exercise/PE02/codebase/model/iris_model.pkl")

# Class mapping for Iris dataset
species = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.route("/")
def home():
    return "<h3>Iris Prediction API is Running</h3>"

@app.route("/health")
def health():
    return jsonify({"status": "OK"})

@app.route("/metadata")
def metadata():
    return jsonify({
        "model_type": "RandomForestClassifier",
        "features": ["sepal length", "sepal width", "petal length", "petal width"],
        "target_classes": list(species.values())
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
      { "features": [sepal_length, sepal_width, petal_length, petal_width] }

    Returns JSON:
      { "prediction": <int>, "species": <string> }
    """
    try:
        data = request.get_json(force=True)

        if not isinstance(data, dict) or "features" not in data:
            return jsonify({"error": "Request JSON must include a 'features' field."}), 400

        feats = data["features"]
        if not isinstance(feats, list) or len(feats) != 4:
            return jsonify({"error": "'features' must be a list of 4 numeric values."}), 400

        try:
            features = np.array([float(x) for x in feats], dtype=float).reshape(1, -1)
        except (TypeError, ValueError):
            return jsonify({"error": "'features' must contain only numeric values."}), 400

        prediction = int(model.predict(features)[0])
        return jsonify({"prediction": prediction, "species": species.get(prediction, "unknown")})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# runtime info
try:
    # Python 3.8+
    from importlib.metadata import version as pkg_version
except Exception:
    pkg_version = None


@app.route("/runtime")
def runtime():
    def safe_version(package_name: str) -> str:
        if pkg_version is None:
            return "unknown"
        try:
            return pkg_version(package_name)
        except Exception:
            return "unknown"

    return jsonify({
        "hostname": socket.gethostname(),
        "packages": {
            "flask": safe_version("flask"),
            "joblib": safe_version("joblib"),
            "scikit-learn": safe_version("scikit-learn"),
        },
        "platform": platform.system(),
        "python_version": sys.version.split()[0]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
