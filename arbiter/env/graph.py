"""Causal Decision Graph Generator for ARBITER.

Generates synthetic AI Decision Systems as causal DAGs.
Nodes: InputFeature, Decision, Outcome, Policy
Edges: Causal, Policy, Temporal
"""
import random
from typing import Optional, Dict, List, Any
import networkx as nx
import numpy as np

# ── Feature catalogue ─────────────────────────────────────────────────────────

EXPLICIT_FEATURES = [
    {"id": "credit_score",      "name": "Credit Score",            "proxy": False, "hidden": False, "protected": False, "values": list(range(300, 851, 50))},
    {"id": "employment_status", "name": "Employment Status",       "proxy": False, "hidden": False, "protected": False, "values": ["employed","self_employed","unemployed","retired"]},
    {"id": "debt_ratio",        "name": "Debt-to-Income Ratio",    "proxy": False, "hidden": False, "protected": False, "values": [round(x*0.05,2) for x in range(20)]},
    {"id": "income",            "name": "Annual Income",           "proxy": False, "hidden": False, "protected": False, "values": list(range(20000, 200001, 10000))},
    {"id": "loan_amount",       "name": "Loan Amount Requested",   "proxy": False, "hidden": False, "protected": False, "values": list(range(5000, 100001, 5000))},
    {"id": "payment_history",   "name": "Payment History Score",   "proxy": False, "hidden": False, "protected": False, "values": list(range(0, 101, 10))},
]

PROXY_FEATURES = [
    {"id": "zip_code_cluster",        "name": "ZIP Code Cluster",              "proxy": True, "hidden": False, "protected": True, "correlates": "race",             "values": [f"cluster_{i}" for i in range(1, 13)]},
    {"id": "surname_pattern",         "name": "Surname Pattern Group",         "proxy": True, "hidden": False, "protected": True, "correlates": "ethnicity",         "values": ["pattern_A","pattern_B","pattern_C","pattern_D"]},
    {"id": "neighborhood_density",    "name": "Neighborhood Commercial Density","proxy": True, "hidden": False, "protected": True, "correlates": "socioeconomic_status","values": ["low","medium","high","very_high"]},
]

HIDDEN_FEATURES = [
    {"id": "internal_risk_score", "name": "Internal Risk Score",   "proxy": False, "hidden": True, "protected": False, "values": list(range(0, 101, 5))},
    {"id": "behavioral_score",    "name": "Behavioral Pattern Score","proxy": False, "hidden": True, "protected": False, "values": list(range(0, 101, 10))},
]

OUTCOME_NODES = [
    {"id": "denial_rate_overall",    "name": "Overall Denial Rate"},
    {"id": "approval_rate_overall",  "name": "Overall Approval Rate"},
    {"id": "flag_rate_overall",      "name": "Overall Flag Rate"},
    {"id": "denial_rate_zip7",       "name": "Denial Rate – ZIP Cluster 7"},
    {"id": "denial_rate_zip3",       "name": "Denial Rate – ZIP Cluster 3"},
    {"id": "approval_rate_majority", "name": "Approval Rate – Majority Demographic"},
    {"id": "denial_rate_minority",   "name": "Denial Rate – Minority Demographic"},
]


def generate_graph(
    seed: Optional[int] = None,
    anomaly_type: Optional[int] = None,
    num_decisions: int = 45,
) -> Dict:
    """
    Generate one episode's causal graph and decision records.

    Returns
    -------
    dict with keys:
        graph            – nx.DiGraph (ground truth, includes hidden nodes)
        observable_graph – nx.DiGraph (Auditor-visible; hidden nodes removed)
        records          – list of decision record dicts
        anomaly_type     – int 1/2/3
        anomaly_info     – ground-truth anomaly metadata
        features         – dict of feature id lists by category
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if anomaly_type is None:
        anomaly_type = random.randint(1, 3)

    G = nx.DiGraph()

    # ── Add feature nodes ─────────────────────────────────────────────────────
    explicit = random.sample(EXPLICIT_FEATURES, 6)
    proxy    = random.sample(PROXY_FEATURES, 3)
    hidden   = random.sample(HIDDEN_FEATURES, 2)

    for feat in explicit + proxy + hidden:
        G.add_node(feat["id"], **feat)

    # ── Add outcome nodes ─────────────────────────────────────────────────────
    for o in OUTCOME_NODES:
        G.add_node(o["id"], id=o["id"], name=o["name"], node_type="outcome")

    # ── Policy node: claims to use only explicit features ────────────────────
    policy_feats = [f["id"] for f in explicit[:3]]  # credit_score, employment_status, debt_ratio
    G.add_node("policy_main", node_type="policy", name="Main Decision Policy", stated_features=policy_feats)
    for fid in policy_feats:
        G.add_edge(fid, "policy_main", edge_type="policy", weight=1.0)
    for fid in policy_feats:
        G.add_edge(fid, "denial_rate_overall", edge_type="policy", weight=1.0)

    # ── Benign causal edges (always present) ─────────────────────────────────
    G.add_edge("credit_score",      "approval_rate_overall",  edge_type="causal", true_causal=True, weight=0.90)
    G.add_edge("debt_ratio",        "denial_rate_overall",    edge_type="causal", true_causal=True, weight=0.85)
    G.add_edge("employment_status", "approval_rate_overall",  edge_type="causal", true_causal=True, weight=0.70)

    # ── Embed anomaly ─────────────────────────────────────────────────────────
    proxy0  = proxy[0]["id"]
    hidden0 = hidden[0]["id"]
    anomaly_info = _embed_anomaly(G, anomaly_type, proxy0, hidden0)

    # ── Decision records ──────────────────────────────────────────────────────
    records = _generate_records(explicit, proxy, hidden, num_decisions, anomaly_type, anomaly_info)
    G.graph["records"]     = records
    G.graph["anomaly_type"] = anomaly_type
    G.graph["anomaly_info"] = anomaly_info

    features = {
        "explicit": [f["id"] for f in explicit],
        "proxy":    [f["id"] for f in proxy],
        "hidden":   [f["id"] for f in hidden],
        "policy_stated": policy_feats,
    }
    G.graph["features"] = features

    observable_G = _make_observable(G)

    return {
        "graph":            G,
        "observable_graph": observable_G,
        "records":          records,
        "anomaly_type":     anomaly_type,
        "anomaly_info":     anomaly_info,
        "features":         features,
    }


def _embed_anomaly(G: nx.DiGraph, anomaly_type: int, proxy0: str, hidden0: str) -> Dict:
    """Embed the true hidden anomaly into the causal graph."""
    info: Dict[str, Any] = {"type": anomaly_type}

    if anomaly_type == 1:
        # Proxy Discrimination: proxy → hidden_risk → denial
        G.add_edge(proxy0,  hidden0,          edge_type="causal", true_causal=True, weight=0.85, label="proxy_drives_risk")
        G.add_edge(hidden0, "denial_rate_overall", edge_type="causal", true_causal=True, weight=0.90, label="risk_drives_denial")
        G.add_edge(hidden0, "denial_rate_minority", edge_type="causal", true_causal=True, weight=0.92)
        info.update({
            "proxy_feature":      proxy0,
            "hidden_mediator":    hidden0,
            "outcome":            "denial_rate_overall",
            "affected_demographic": f"{proxy0}_7",
            "recommended_action": "retrain",
            "causal_chain":       [proxy0, hidden0, "denial_rate_overall"],
        })

    elif anomaly_type == 2:
        # Adversarial Injection: manipulated records leave statistical fingerprints
        G.add_edge(proxy0, "denial_rate_minority", edge_type="causal", true_causal=True, weight=0.75, label="injection_fingerprint")
        info.update({
            "injected_feature":    proxy0,
            "injection_value":     "cluster_7",
            "injection_rate":      0.12,
            "affected_demographic": f"{proxy0}_7",
            "recommended_action":  "audit",
            "causal_chain":        [proxy0, "denial_rate_minority"],
        })

    elif anomaly_type == 3:
        # Model Drift: pre-drift uses credit_score, post-drift silently switches to proxy
        drift_ts = round(random.uniform(0.4, 0.6), 2)
        G.add_edge("credit_score", "denial_rate_overall", edge_type="causal", true_causal=True, weight=0.88, temporal_scope="pre_drift")
        G.add_edge(proxy0, "denial_rate_overall",         edge_type="causal", true_causal=True, weight=0.84, temporal_scope="post_drift")
        G.add_edge(proxy0, hidden0,                        edge_type="temporal", drift_point=drift_ts, weight=0.70)
        info.update({
            "drift_timestamp":    drift_ts,
            "pre_drift_cause":    "credit_score",
            "post_drift_cause":   proxy0,
            "drift_mediator":     hidden0,
            "affected_demographic": f"{proxy0}_7",
            "recommended_action": "retrain",
            "causal_chain":       [proxy0, hidden0, "denial_rate_overall"],
        })

    return info


def _generate_records(explicit, proxy, hidden, n, anomaly_type, anomaly_info) -> List[Dict]:
    """Generate n synthetic decision records for the episode."""
    records = []
    drift_ts = anomaly_info.get("drift_timestamp", 0.5)

    for i in range(n):
        ts = i / n
        fvec  = {f["id"]: random.choice(f["values"]) for f in explicit}
        pvec  = {f["id"]: random.choice(f["values"]) for f in proxy}
        hvec  = {f["id"]: random.choice(f["values"]) for f in hidden}

        is_anomalous = False
        is_injected  = False

        if anomaly_type == 1:
            if pvec.get(proxy[0]["id"], "") == "cluster_7":
                outcome = "denied"; is_anomalous = True
            else:
                outcome = "approved" if fvec.get("credit_score", 600) > 620 else "denied"

        elif anomaly_type == 2:
            if i < int(n * 0.12):
                pvec[proxy[0]["id"]] = "cluster_7"
                outcome = "denied"; is_anomalous = True; is_injected = True
            else:
                outcome = "approved" if fvec.get("credit_score", 600) > 620 else "denied"

        elif anomaly_type == 3:
            if ts < drift_ts:
                outcome = "approved" if fvec.get("credit_score", 600) > 620 else "denied"
            else:
                if pvec.get(proxy[0]["id"], "") == "cluster_7":
                    outcome = "denied"; is_anomalous = True
                else:
                    outcome = "approved" if fvec.get("credit_score", 600) > 620 else "denied"
        else:
            outcome = "approved" if fvec.get("credit_score", 600) > 620 else "denied"

        records.append({
            "id":             f"rec_{i:04d}",
            "feature_vector": fvec,
            "proxy_vector":   pvec,
            "hidden_vector":  hvec,   # never shown to Auditor
            "outcome":        outcome,
            "timestamp":      ts,
            "is_anomalous":   is_anomalous,
            "is_injected":    is_injected,
        })

    return records


def _make_observable(G: nx.DiGraph) -> nx.DiGraph:
    """Remove hidden feature nodes to produce what the Auditor can observe."""
    obs = G.copy()
    hidden_nodes = [n for n, d in G.nodes(data=True) if d.get("hidden")]
    obs.remove_nodes_from(hidden_nodes)
    return obs
