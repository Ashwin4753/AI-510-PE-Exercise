import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.pkl")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Model trained. Test accuracy: {acc:.4f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()