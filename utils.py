import os
import json
import datetime

def log_simulation(service, message):
    """Simulates Azure portal logs for resource operations."""
    print(f"[AZURE-SIMULATION] {service}: {message}")

def ensure_workspace():
    """Creates a local JSON file to simulate an Azure ML workspace."""
    if not os.path.exists("workspace.json"):
        ws = {
            "workspace_name": "azure-ml-demo",
            "location": "westus",
            "created_at": datetime.datetime.now().isoformat()
        }
        with open("workspace.json", "w") as f:
            json.dump(ws, f, indent=2)
        log_simulation("AzureML Workspace", "Workspace created.")
    else:
        log_simulation("AzureML Workspace", "Workspace already exists.")