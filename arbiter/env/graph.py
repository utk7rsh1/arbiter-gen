"""Causal Decision Graph Generator for ARBITER.

Generates synthetic AI Decision Systems as causal DAGs.
Nodes: InputFeature, Decision, Outcome, Policy
Edges: Causal, Policy, Temporal

Supports two modes:
  - Loan domain (backwards-compatible default): pass no `domain` argument.
  - Any domain: pass a DomainConfig produced by GroqGraphGenerator.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

# ── Loan-domain feature catalogue (used when domain=None) ─────────────────────

EXPLICIT_FEATURES = [
    {"id": "credit_score",      "name": "Credit Score",             "proxy": False, "hidden": False, "protected": False, "values": list(range(300, 851, 50))},
    {"id": "employment_status", "name": "Employment Status",        "proxy": False, "hidden": False, "protected": False, "values": ["employed", "self_employed", "unemployed", "retired"]},
    {"id": "debt_ratio",        "name": "Debt-to-Income Ratio",     "proxy": False, "hidden": False, "protected": False, "values": [round(x * 0.05, 2) for x in range(20)]},
    {"id": "income",            "name": "Annual Income",            "proxy": False, "hidden": False, "protected": False, "values": list(range(20000, 200001, 10000))},
    {"id": "loan_amount",       "name": "Loan Amount Requested",    "proxy": False, "hidden": False, "protected": False, "values": list(range(5000, 100001, 5000))},
    {"id": "payment_history",   "name": "Payment History Score",    "proxy": False, "hidden": False, "protected": False, "values": list(range(0, 101, 10))},
]

PROXY_FEATURES = [
    {"id": "zip_code_cluster",     "name": "ZIP Code Cluster",               "proxy": True, "hidden": False, "protected": True, "correlates": "race",               "values": [f"cluster_{i}" for i in range(1, 13)]},
    {"id": "surname_pattern",      "name": "Surname Pattern Group",          "proxy": True, "hidden": False, "protected": True, "correlates": "ethnicity",           "values": ["pattern_A", "pattern_B", "pattern_C", "pattern_D"]},
    {"id": "neighborhood_density", "name": "Neighborhood Commercial Density","proxy": True, "hidden": False, "protected": True, "correlates": "socioeconomic_status","values": ["low", "medium", "high", "very_high"]},
]

HIDDEN_FEATURES = [
    {"id": "internal_risk_score", "name": "Internal Risk Score",      "proxy": False, "hidden": True, "protected": False, "values": list(range(0, 101, 5))},
    {"id": "behavioral_score",    "name": "Behavioral Pattern Score", "proxy": False, "hidden": True, "protected": False, "values": list(range(0, 101, 10))},
]

OUTCOME_NODES = [
    {"id": "denial_rate_overall",    "name": "Overall Denial Rate"},
    {"id": "approval_rate_overall",  "name": "Overall Approval Rate"},
    {"id": "flag_rate_overall",      "name": "Overall Flag Rate"},
    {"id": "denial_rate_zip7",       "name": "Denial Rate - ZIP Cluster 7"},
    {"id": "denial_rate_zip3",       "name": "Denial Rate - ZIP Cluster 3"},
    {"id": "approval_rate_majority", "name": "Approval Rate - Majority Demographic"},
    {"id": "denial_rate_minority",   "name": "Denial Rate - Minority Demographic"},
]


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_graph(
    seed: Optional[int] = None,
    anomaly_type: Optional[int] = None,
    num_decisions: int = 45,
    domain: Optional[Any] = None,  # DomainConfig | None
) -> Dict:
    """Generate one episode's causal graph and decision records.

    Parameters
    ----------
    seed          : RNG seed for reproducibility.
    anomaly_type  : 1 = proxy discrimination, 2 = adversarial injection,
                    3 = model drift.  None -> random.
    num_decisions : Number of synthetic decision records.
    domain        : DomainConfig produced by GroqGraphGenerator.
                    None -> use hardcoded loan-domain constants (backwards compat).

    Returns
    -------
    dict with keys:
        graph            - nx.DiGraph (ground truth, includes hidden nodes)
        observable_graph - nx.DiGraph (Auditor-visible; hidden nodes removed)
        records          - list of decision record dicts
        anomaly_type     - int 1/2/3
        anomaly_info     - ground-truth anomaly metadata
        features         - dict of feature id lists by category
        domain           - the DomainConfig used (or None for loan domain)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if anomaly_type is None:
        anomaly_type = random.randint(1, 3)

    # ── Adapter block: domain -> internal constants ───────────────────────────
    if domain is not None:
        explicit = [f.to_graph_dict() for f in domain.explicit_features]
        proxy    = [f.to_graph_dict(is_proxy=True) for f in domain.proxy_features]
        hidden   = [f.to_graph_dict(is_hidden=True) for f in domain.hidden_features]

        # Honor outcome_nodes verbatim if declared, otherwise build from schema
        if domain.outcome_nodes:
            # Build dicts from the declared string names; ensure required IDs exist.
            raw_ids = list(domain.outcome_nodes)
            outcome_nodes = [{"id": n, "name": n.replace("_", " ").title()} for n in raw_ids]
            # Ensure the three required outcome IDs are present
            required = {
                domain.denial_outcome_id():    f"Overall {domain.negative_outcome.title()} Rate",
                domain.approval_outcome_id():  f"Overall {domain.positive_outcome.title()} Rate",
                domain.minority_outcome_id():  f"{domain.negative_outcome.title()} Rate - Minority",
            }
            existing_ids = {n["id"] for n in outcome_nodes}
            for oid, oname in required.items():
                if oid not in existing_ids:
                    outcome_nodes.append({"id": oid, "name": oname})
        else:
            outcome_nodes = domain.build_outcome_nodes()

        DENIAL_RATE_OVERALL   = domain.denial_outcome_id()
        APPROVAL_RATE_OVERALL = domain.approval_outcome_id()
        DENIAL_RATE_MINORITY  = domain.minority_outcome_id()
        THRESHOLD_FEATURE     = domain.approval_threshold_feature
        THRESHOLD_VALUE       = domain.approval_threshold_value
        DISCRIMINATED_VAL     = domain.discriminated_group_value
        POSITIVE_OUTCOME      = domain.positive_outcome
        NEGATIVE_OUTCOME      = domain.negative_outcome

        # Resolve proxy0 by discriminated_group_feature NAME (not index)
        proxy_name_map = {f["id"]: f for f in proxy}
        proxy0_id = domain.discriminated_group_feature
        if proxy0_id not in proxy_name_map:
            # Fallback: try causal_chain[0], then index 0
            proxy0_id = domain.causal_chain[0] if domain.causal_chain else proxy[0]["id"]
            if proxy0_id not in proxy_name_map:
                proxy0_id = proxy[0]["id"]

        # Resolve hidden0 by causal_chain[1] NAME (not index)
        hidden_name_map = {f["id"]: f for f in hidden}
        hidden0_id = domain.causal_chain[1] if len(domain.causal_chain) > 1 else hidden[0]["id"]
        if hidden0_id not in hidden_name_map:
            hidden0_id = hidden[0]["id"]

        # Resolve pre_drift_cause as approval_threshold_feature (better signal)
        explicit_name_map = {f["id"]: f for f in explicit}
        pre_drift_cause = THRESHOLD_FEATURE
        if pre_drift_cause not in explicit_name_map:
            pre_drift_cause = explicit[0]["id"]

        # Build domain_context stash for downstream modules
        domain_context = {
            "discriminated_value":              DISCRIMINATED_VAL,
            "discriminated_feature":            proxy0_id,
            "threshold_feature":                THRESHOLD_FEATURE,
            "threshold_value":                  THRESHOLD_VALUE,
            "positive_outcome":                 POSITIVE_OUTCOME,
            "negative_outcome":                 NEGATIVE_OUTCOME,
            "denial_outcome":                   DENIAL_RATE_OVERALL,
            "approval_outcome":                 APPROVAL_RATE_OVERALL,
            "minority_outcome":                 DENIAL_RATE_MINORITY,
            "seasonal_decoy_feature":           getattr(domain, "seasonal_decoy_feature", None),
            "seasonal_decoy_secondary_feature": getattr(domain, "seasonal_decoy_secondary_feature", None),
            "legitimate_risk_feature":          getattr(domain, "legitimate_risk_feature", None),
            "legitimate_risk_threshold":        getattr(domain, "legitimate_risk_threshold", None),
            "drift_regulation_name":            getattr(domain, "drift_regulation_name", None),
            "newly_non_compliant_features":     getattr(domain, "newly_non_compliant_features", None),
            "drift_change_description":         getattr(domain, "drift_change_description", None),
        }
    else:
        explicit = random.sample(EXPLICIT_FEATURES, 6)
        proxy    = random.sample(PROXY_FEATURES, 3)
        hidden   = random.sample(HIDDEN_FEATURES, 2)
        outcome_nodes = OUTCOME_NODES

        DENIAL_RATE_OVERALL   = "denial_rate_overall"
        APPROVAL_RATE_OVERALL = "approval_rate_overall"
        DENIAL_RATE_MINORITY  = "denial_rate_minority"
        THRESHOLD_FEATURE     = "credit_score"
        THRESHOLD_VALUE       = 620
        DISCRIMINATED_VAL     = "cluster_7"
        POSITIVE_OUTCOME      = "approved"
        NEGATIVE_OUTCOME      = "denied"

        proxy0_id       = proxy[0]["id"]
        hidden0_id      = hidden[0]["id"]
        pre_drift_cause = explicit[0]["id"]

        domain_context = None   # loan path: downstream modules use their own fallbacks

    G = nx.DiGraph()

    # ── Add feature nodes ──────────────────────────────────────────────────────
    for feat in explicit + proxy + hidden:
        G.add_node(feat["id"], **feat)

    # ── Add outcome nodes ──────────────────────────────────────────────────────
    for o in outcome_nodes:
        G.add_node(o["id"], id=o["id"], name=o["name"], node_type="outcome")

    # ── Policy node: claims to use only the first 3 explicit features ─────────
    policy_feats = [f["id"] for f in explicit[:3]]
    G.add_node(
        "policy_main",
        node_type="policy",
        name="Main Decision Policy",
        stated_features=policy_feats,
    )
    for fid in policy_feats:
        G.add_edge(fid, "policy_main", edge_type="policy", weight=1.0)
    for fid in policy_feats:
        G.add_edge(fid, DENIAL_RATE_OVERALL, edge_type="policy", weight=1.0)

    # ── Benign causal edges (always present) ──────────────────────────────────
    # Use THRESHOLD_FEATURE for the approval edge (correct signal);
    # pick the next two explicit features for the other benign edges.
    threshold_feat_id = THRESHOLD_FEATURE
    other_explicit = [f["id"] for f in explicit if f["id"] != threshold_feat_id]
    benign_feat1 = other_explicit[0] if other_explicit else explicit[0]["id"]
    benign_feat2 = other_explicit[1] if len(other_explicit) > 1 else benign_feat1

    G.add_edge(threshold_feat_id, APPROVAL_RATE_OVERALL, edge_type="causal", true_causal=True, weight=0.90)
    G.add_edge(benign_feat1,      DENIAL_RATE_OVERALL,   edge_type="causal", true_causal=True, weight=0.85)
    G.add_edge(benign_feat2,      APPROVAL_RATE_OVERALL, edge_type="causal", true_causal=True, weight=0.70)

    # ── Embed anomaly ──────────────────────────────────────────────────────────
    anomaly_info = _embed_anomaly(
        G,
        anomaly_type,
        proxy0_id,
        hidden0_id,
        denial_outcome=DENIAL_RATE_OVERALL,
        minority_outcome=DENIAL_RATE_MINORITY,
        pre_drift_cause=pre_drift_cause,
        discriminated_val=DISCRIMINATED_VAL,
    )

    # ── Decision records ───────────────────────────────────────────────────────
    records = _generate_records(
        explicit,
        proxy,
        hidden,
        num_decisions,
        anomaly_type,
        anomaly_info,
        proxy0_id=proxy0_id,
        threshold_feature=THRESHOLD_FEATURE,
        threshold_value=THRESHOLD_VALUE,
        discriminated_val=DISCRIMINATED_VAL,
        positive_outcome=POSITIVE_OUTCOME,
        negative_outcome=NEGATIVE_OUTCOME,
    )
    G.graph["records"]      = records
    G.graph["anomaly_type"] = anomaly_type
    G.graph["anomaly_info"] = anomaly_info

    features = {
        "explicit":      [f["id"] for f in explicit],
        "proxy":         [f["id"] for f in proxy],
        "hidden":        [f["id"] for f in hidden],
        "policy_stated": policy_feats,
    }
    G.graph["features"]       = features
    G.graph["domain_context"] = domain_context   # None for loan path

    observable_G = _make_observable(G)

    return {
        "graph":            G,
        "observable_graph": observable_G,
        "records":          records,
        "anomaly_type":     anomaly_type,
        "anomaly_info":     anomaly_info,
        "features":         features,
        "domain":           domain,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _embed_anomaly(
    G: nx.DiGraph,
    anomaly_type: int,
    proxy0: str,
    hidden0: str,
    denial_outcome: str = "denial_rate_overall",
    minority_outcome: str = "denial_rate_minority",
    pre_drift_cause: str = "credit_score",
    discriminated_val: str = "cluster_7",
) -> Dict:
    """Embed the true hidden anomaly into the causal graph."""
    info: Dict[str, Any] = {"type": anomaly_type}

    if anomaly_type == 1:
        G.add_edge(proxy0,  hidden0,         edge_type="causal", true_causal=True, weight=0.85, label="proxy_drives_risk")
        G.add_edge(hidden0, denial_outcome,  edge_type="causal", true_causal=True, weight=0.90, label="risk_drives_denial")
        G.add_edge(hidden0, minority_outcome,edge_type="causal", true_causal=True, weight=0.92)
        info.update({
            "proxy_feature":       proxy0,
            "hidden_mediator":     hidden0,
            "outcome":             denial_outcome,
            "affected_demographic": f"{proxy0}_{discriminated_val}",
            "recommended_action":  "retrain",
            "causal_chain":        [proxy0, hidden0, denial_outcome],
        })

    elif anomaly_type == 2:
        G.add_edge(proxy0, minority_outcome, edge_type="causal", true_causal=True, weight=0.75, label="injection_fingerprint")
        info.update({
            "injected_feature":     proxy0,
            "injection_value":      discriminated_val,
            "injection_rate":       0.12,
            "affected_demographic": f"{proxy0}_{discriminated_val}",
            "recommended_action":   "audit",
            "causal_chain":         [proxy0, minority_outcome],
        })

    elif anomaly_type == 3:
        drift_ts = round(random.uniform(0.4, 0.6), 2)
        G.add_edge(pre_drift_cause, denial_outcome, edge_type="causal", true_causal=True, weight=0.88, temporal_scope="pre_drift")
        G.add_edge(proxy0,          denial_outcome, edge_type="causal", true_causal=True, weight=0.84, temporal_scope="post_drift")
        G.add_edge(proxy0,          hidden0,        edge_type="temporal", drift_point=drift_ts, weight=0.70)
        info.update({
            "drift_timestamp":     drift_ts,
            "pre_drift_cause":     pre_drift_cause,
            "post_drift_cause":    proxy0,
            "drift_mediator":      hidden0,
            "affected_demographic": f"{proxy0}_{discriminated_val}",
            "recommended_action":  "retrain",
            "causal_chain":        [proxy0, hidden0, denial_outcome],
        })

    return info


def _generate_records(
    explicit: List[Dict],
    proxy: List[Dict],
    hidden: List[Dict],
    n: int,
    anomaly_type: int,
    anomaly_info: Dict,
    proxy0_id: Optional[str] = None,
    threshold_feature: str = "credit_score",
    threshold_value: float = 620,
    discriminated_val: str = "cluster_7",
    positive_outcome: str = "approved",
    negative_outcome: str = "denied",
) -> List[Dict]:
    """Generate n synthetic decision records for the episode."""
    records  = []
    drift_ts = anomaly_info.get("drift_timestamp", 0.5)

    # Determine which proxy feature carries the anomaly
    if proxy0_id is None:
        proxy0_id = proxy[0]["id"]

    for i in range(n):
        ts   = i / n
        fvec = {f["id"]: random.choice(f["values"]) for f in explicit}
        pvec = {f["id"]: random.choice(f["values"]) for f in proxy}
        hvec = {f["id"]: random.choice(f["values"]) for f in hidden}

        is_anomalous = False
        is_injected  = False

        def _threshold_outcome() -> str:
            val = fvec.get(threshold_feature)
            if val is not None and isinstance(val, (int, float)) and val > threshold_value:
                return positive_outcome
            return negative_outcome

        if anomaly_type == 1:
            if pvec.get(proxy0_id, "") == discriminated_val:
                outcome = negative_outcome
                is_anomalous = True
            else:
                outcome = _threshold_outcome()

        elif anomaly_type == 2:
            if i < int(n * 0.12):
                pvec[proxy0_id] = discriminated_val
                outcome      = negative_outcome
                is_anomalous = True
                is_injected  = True
            else:
                outcome = _threshold_outcome()

        elif anomaly_type == 3:
            if ts < drift_ts:
                outcome = _threshold_outcome()
            else:
                if pvec.get(proxy0_id, "") == discriminated_val:
                    outcome      = negative_outcome
                    is_anomalous = True
                else:
                    outcome = _threshold_outcome()

        else:
            outcome = _threshold_outcome()

        records.append({
            "id":             f"rec_{i:04d}",
            "feature_vector": fvec,
            "proxy_vector":   pvec,
            "hidden_vector":  hvec,   # never shown to the Auditor
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
