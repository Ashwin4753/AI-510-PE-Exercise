import os
import datetime
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import log_simulation, ensure_workspace

# Simulate workspace setup
ensure_workspace()

# Step 1: Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Step 2: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 3: Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))

# Step 4: Save locally (simulate Azure Blob)
os.makedirs("azure_storage/models", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"azure_storage/models/iris_model_{timestamp}.pkl"
joblib.dump(model, model_path)
log_simulation("Azure Blob", f"Uploaded model artifact to {model_path}")

# Step 5: Register model (simulate Azure ML registry)
entry = {"timestamp": timestamp, "model": model_path, "accuracy": round(accuracy, 3)}
df = pd.DataFrame([entry])
header = not os.path.exists("model_registry.csv")
df.to_csv("model_registry.csv", mode="a", header=header, index=False)
log_simulation("Azure ML Registry", f"Model registered with accuracy {accuracy:.2f}")