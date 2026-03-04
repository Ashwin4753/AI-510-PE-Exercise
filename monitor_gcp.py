import random
from utils_gcp import gcp_log

def monitor_model():
    latency = round(random.uniform(20, 100), 2)
    accuracy = round(random.uniform(0.80, 0.99), 2)
    gcp_log("Cloud Monitoring", f"Latency: {latency} ms | Accuracy: {accuracy}")

if __name__ == "__main__":
    gcp_log("Cloud Monitoring", "Starting model monitoring simulation...")
    monitor_model()