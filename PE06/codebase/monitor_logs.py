import datetime
import random
from utils import get_cloudwatch_client, dry_run_log

# Step 1: Initialize simulated CloudWatch client
cw = get_cloudwatch_client()

# Step 2: Simulate fetching metrics from an AWS SageMaker endpoint
dry_run_log("CloudWatch", "Fetching model inference metrics")

# Simulated metric data
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
requests = 150
latency = f"{random.randint(40, 60)}ms"
error_rate = "0.5%"

print("[CloudWatch Simulation]")
print(f"Timestamp: {timestamp}")
print(f"Requests served: {requests}")
print(f"Average latency: {latency}")
print(f"Error rate: {error_rate}")