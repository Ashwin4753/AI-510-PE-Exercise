from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (update path if yours differs)
model = joblib.load("/workspaces/AI-510-PE-Exercise/PE01/Codebase/model/iris_model.pkl")

# Map numeric prediction -> species name
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.route('/')
def index():
    return "MLOps Flask API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Optional safety check (helps avoid errors on bad input)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = np.array(data["features"]).reshape(1, -1)

    pred_label = int(model.predict(features)[0])
    pred_species = species_map.get(pred_label, "unknown")

    return jsonify({
        "prediction": pred_label,
        "species": pred_species
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)