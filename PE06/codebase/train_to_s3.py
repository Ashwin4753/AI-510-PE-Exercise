import os
import joblib
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from utils import get_s3_client, dry_run_log

#Step 1: Train model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

#Step 1b: Compute accuracy (simple baseline on training set)
accuracy = model.score(X, y)
print("Model trained successfully.")
print(f"Training accuracy: {accuracy:.4f}")

#Step 2: Create a timestamped version folder (simulates model versioning)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"iris_model_{timestamp}"
local_dir = os.path.join("model", model_name)
os.makedirs(local_dir, exist_ok=True)

#Step 3: Save model locally inside version folder
model_path = os.path.join(local_dir, "iris_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

#Step 4: Simulate S3 upload to a versioned key
s3 = get_s3_client()
bucket_name = "mlops-demo-bucket"
s3_key = f"models/{model_name}/iris_model.pkl"
dry_run_log("S3", f"Uploading {model_path} to s3://{bucket_name}/{s3_key}")
print(f"Simulated upload to s3://{bucket_name}/{s3_key}")

#Step 5: Append metadata to local registry log
registry_line = f"{model_name} | timestamp={timestamp} | accuracy={accuracy:.4f}\n"
with open("model_registry.log", "a", encoding="utf-8") as f:
    f.write(registry_line)
print("Logged model metadata to model_registry.log")