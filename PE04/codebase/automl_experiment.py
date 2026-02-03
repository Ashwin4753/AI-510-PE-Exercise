from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define candidate models
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

# AutoML-style model selection
results = []
for name, model in models.items():
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    results.append({"model": name, "score": score})
    print(f"{name}: {score:.4f}")

# Find and save the best model
results_df = pd.DataFrame(results)
best = results_df.loc[results_df["score"].idxmax()]
print(f"\nBest Model: {best['model']} ({best['score']:.4f})")

final_model = models[best["model"]].fit(X_train, y_train)
joblib.dump(final_model, "model/best_model.pkl")
results_df.to_csv("model/performance_log.csv", index=False)

print("Model saved to model/best_model.pkl")