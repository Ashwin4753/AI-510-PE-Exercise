import os
import pickle
import random
import numpy as np
import onnxruntime as ort

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Load dataset
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target

    # Train Scikit-learn model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Save Scikit-learn model
    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Convert model to ONNX
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save ONNX model
    with open("models/iris_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Model converted to ONNX successfully.")

    # Load ONNX model with ONNX Runtime
    session = ort.InferenceSession("models/iris_model.onnx")
    input_name = session.get_inputs()[0].name

    # HOS09A: prediction comparison
    print("\nPrediction Comparison")
    for i in range(10):
        sample = X[i:i + 1]
        sklearn_pred = model.predict(sample)[0]
        onnx_pred = session.run(None, {input_name: sample})[0][0]
        print(f"Sample {i}: sklearn → {sklearn_pred}, onnx → {onnx_pred}")

    # PE09: random 10 sample check
    random_indices = random.sample(range(len(X)), 10)
    mismatch_count = 0

    print("\nMismatch Check on 10 Random Samples")
    for idx in random_indices:
        sample = X[idx:idx + 1]
        sklearn_pred = model.predict(sample)[0]
        onnx_pred = session.run(None, {input_name: sample})[0][0]

        if sklearn_pred != onnx_pred:
            print(f"Mismatch at Sample {idx}: sklearn → {sklearn_pred}, onnx → {onnx_pred}")
            mismatch_count += 1

    print(f"Total mismatches: {mismatch_count}")

if __name__ == "__main__":
    main()