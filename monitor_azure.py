import random
from utils import log_simulation

def simulate_metrics():
    latency = round(random.uniform(80, 150), 2)
    error_rate = round(random.uniform(0.01, 0.05), 3)
    log_simulation("Azure Monitor", f"Latency={latency}ms, Error Rate={error_rate}")
    return latency, error_rate

if __name__ == "__main__":
    simulate_metrics()