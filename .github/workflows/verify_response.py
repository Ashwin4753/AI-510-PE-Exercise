import json

with open("response.json", "r", encoding="utf-8") as f:
    resp = json.load(f)

# Required fields
assert "prediction" in resp, "Missing field: prediction"
assert "probabilities" in resp, "Missing field: probabilities"

# Validate prediction
pred = resp["prediction"]
assert isinstance(pred, int), f"prediction must be int, got {type(pred)}"
assert pred in [0, 1, 2], f"prediction out of range: {pred}"

# Validate probabilities
probs = resp["probabilities"]
assert isinstance(probs, list) and len(probs) == 3, "probabilities must be list of length 3"
assert all(isinstance(x, (int, float)) for x in probs), "probabilities must be numeric"

s = sum(probs)
assert abs(s - 1.0) < 0.10, f"probabilities should sum ~1.0, got {s}"

print("✅ PE03 integration test passed")
