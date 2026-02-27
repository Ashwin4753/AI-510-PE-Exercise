from utils import log_simulation
import pandas as pd

def deploy_latest_model():
    df = pd.read_csv("model_registry.csv")
    latest = df.iloc[-1]
    log_simulation("Azure ML Deployment", f"Deploying model {latest['model']} (accuracy {latest['accuracy']})")
    print(f"Model deployed successfully at {latest['timestamp']}")

if __name__ == "__main__":
    deploy_latest_model()