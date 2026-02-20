import boto3
from unittest.mock import MagicMock

def get_s3_client():
    """Simulate S3 client connection."""
    mock_s3 = MagicMock()
    return mock_s3

def get_sagemaker_client():
    """Simulate SageMaker client."""
    mock_sm = MagicMock()
    return mock_sm

def get_cloudwatch_client():
    """Simulate CloudWatch client."""
    mock_cw = MagicMock()
    return mock_cw

def dry_run_log(service, action):
    print(f"[SIMULATION] {service}: {action}")