from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os
from utils_gcp import gcp_log, ensure_bucket

ensure_bucket()
iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)
gcp_log("Vertex AI", "Training completed on local data.")

os.makedirs("gcp_bucket/models", exist_ok=True)
joblib.dump(model, "gcp_bucket/models/iris_model.pkl")
gcp_log("GCS", "Model uploaded to simulated GCS bucket.")