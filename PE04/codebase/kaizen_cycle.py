import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import os

# Load dataset and simulate new data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=np.random.randint(100)
)

# Load the current best model
old_model = joblib.load("model/best_model.pkl")
old_acc = accuracy_score(y_test, old_model.predict(X_test))

# Train a new model with different parameters (Kaizen step)
new_model = RandomForestClassifier(
    n_estimators=150,
    random_state=np.random.randint(100)
)
new_model.fit(X_train, y_train)
new_acc = accuracy_score(y_test, new_model.predict(X_test))

print(f"Old Accuracy: {old_acc:.3f}, New Accuracy: {new_acc:.3f}")

# Decide whether to replace model
improved = new_acc > old_acc
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if improved:
    joblib.dump(new_model, "model/best_model.pkl")
    action = "MODEL_UPDATED"
    print("Model improved and updated.")
else:
    action = "MODEL_KEPT"
    print("New model not better. Keeping the previous one.")

# Ensure log file exists
log_path = "model/performance_log.csv"
if os.path.exists(log_path):
    log = pd.read_csv(log_path)
else:
    log = pd.DataFrame(
        columns=[
            "timestamp",
            "old_accuracy",
            "new_accuracy",
            "improved",
            "action"
        ]
    )

# Append PE04 traceability log
new_log_entry = {
    "timestamp": timestamp,
    "old_accuracy": round(old_acc, 4),
    "new_accuracy": round(new_acc, 4),
    "improved": "YES" if improved else "NO",
    "action": action
}

log = pd.concat([log, pd.DataFrame([new_log_entry])], ignore_index=True)
log.to_csv(log_path, index=False)

print("Performance log updated.")