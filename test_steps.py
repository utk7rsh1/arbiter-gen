import requests

BASE = "http://localhost:8000"

# Create + reset
s = requests.post(f"{BASE}/sessions", json={"level": 1, "seed": 42})
print("Create:", s.status_code, s.json())
sid = s.json()["session_id"]

r = requests.post(f"{BASE}/sessions/{sid}/reset", json={"seed": 42})
print("Reset:", r.status_code)

def step(label, action_dict):
    r = requests.post(f"{BASE}/sessions/{sid}/step", json={"action": action_dict})
    if r.status_code == 200:
        d = r.json()
        print(f"  OK  {label:40s} step={d['observation']['step']}  reward={d['reward']}")
    else:
        print(f"  ERR {label:40s} {r.status_code}: {r.text[:300]}")

step("QUERY_RECORDS", {"type": "QUERY_RECORDS", "feature_filter": {}, "outcome_filter": "denied"})
step("QUERY_FEATURE_DISTRIBUTION", {"type": "QUERY_FEATURE_DISTRIBUTION", "feature_id": "zip_code_cluster", "group_by": "loan_outcome"})
step("CLAIM_CAUSAL", {"type": "CLAIM_CAUSAL", "claim": {
    "cause_feature": "zip_code_cluster",
    "effect_outcome": "denial_rate_overall",
    "mechanism": "internal_risk_score",
    "direction": "positive",
    "confidence": "HIGH",
    "basis_records": ["rec_001", "rec_007", "rec_012"],
    "anomaly_type": "proxy_discrimination",
}})
step("QUERY_COUNTERFACTUAL", {"type": "QUERY_COUNTERFACTUAL", "record_id": "rec_001", "feature_id": "zip_code_cluster", "counterfactual_value": 3})
step("CLAIM_COUNTERFACTUAL", {"type": "CLAIM_COUNTERFACTUAL", "claim": {
    "subject_record": "rec_001",
    "counterfactual_feature": "zip_code_cluster",
    "predicted_outcome_change": "approved",
    "confidence": "HIGH",
    "basis": "cf_query_step_3",
}})
step("FLAG_HYPOTHESIS confirmed", {"type": "FLAG_HYPOTHESIS", "hypothesis_type": "proxy_discrimination", "status": "CONFIRMED"})
step("CLAIM_THEORY_OF_MIND", {"type": "CLAIM_THEORY_OF_MIND", "claim": {
    "defender_action": "obfuscating",
    "target_link": "zip_code_cluster->internal_risk_score",
    "obfuscation_method": "timestamp_manipulation",
    "confidence": "HIGH",
    "basis": "timestamp_clustering",
}})
step("SUBMIT_REPORT", {"type": "SUBMIT_REPORT",
    "anomaly_type": "proxy_discrimination",
    "primary_evidence_chain": ["zip_code_cluster", "internal_risk_score", "denial_rate_overall"],
    "affected_demographic": "zip_code_cluster_7",
    "recommended_action": "audit_risk_score_model",
})
