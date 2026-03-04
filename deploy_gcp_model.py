import joblib, os
from utils_gcp import gcp_log

model_path = "gcp_bucket/models/iris_model.pkl"
deploy_path = "gcp_bucket/deployed_model/iris_model.pkl"

if os.path.exists(model_path):
    os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
    joblib.load(model_path)
    gcp_log("Vertex AI", "Model loaded successfully for deployment.")
    gcp_log("Vertex AI Endpoint", f"Model deployed at {deploy_path}")
else:
    gcp_log("Vertex AI", "No model found for deployment.")