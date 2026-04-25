"""Decoy Generation Functions for ARBITER.

Two decoys are always present per episode:
  Decoy A - Seasonal variation that superficially looks like model drift.
  Decoy B - A legitimate risk-based feature that superficially looks like proxy discrimination.

Both require 2-3 queries to eliminate conclusively.

Domain-specific features are read from the optional `domain_context` dict
(set by graph.py on G.graph["domain_context"]).  When absent, the original
loan-domain feature names are used so the existing loan path is unchanged.
"""
import random
from typing import Dict, List, Optional


def generate_decoys(
    records: List[Dict],
    features: Dict,
    domain_context: Optional[Dict] = None,
) -> Dict:
    """
    Generate two decoys and inject them into the records/graph metadata.

    Parameters
    ----------
    records        : Decision records list from generate_graph().
    features       : Feature id lists dict from generate_graph().
    domain_context : Optional dict from G.graph["domain_context"].
                     When None, loan-domain feature names are used.

    Returns
    -------
    dict with keys:
        decoy_a  - seasonal variation decoy metadata
        decoy_b  - legitimate-risk decoy metadata
        records  - updated records list (with decoy markers)
    """
    records = [r.copy() for r in records]
    n = len(records)

    decoy_a = _generate_seasonal_variation_decoy(records, n, domain_context)
    decoy_b = _generate_legitimate_risk_decoy(records, features, domain_context)

    return {
        "decoy_a": decoy_a,
        "decoy_b": decoy_b,
        "records": records,
    }


def _generate_seasonal_variation_decoy(
    records: List[Dict],
    n: int,
    domain_context: Optional[Dict] = None,
) -> Dict:
    """
    Decoy A: Approval rates drop mid-episode (looks like model drift).
    True explanation: seasonal demand surge, not a policy change.

    Detection: QUERY_FEATURE_DISTRIBUTION on the seasonal feature over time
               reveals higher values in the mid-period (demand, not bias).
    Elimination: 2 queries needed (distribution + counterfactual).
    """
    # Resolve feature names from context or fall back to loan literals
    if domain_context:
        primary_feat   = domain_context.get("seasonal_decoy_feature")   or "loan_amount"
        secondary_feat = domain_context.get("seasonal_decoy_secondary_feature") or "debt_ratio"
        negative_outcome = domain_context.get("negative_outcome", "denied")
    else:
        primary_feat   = "loan_amount"
        secondary_feat = "debt_ratio"
        negative_outcome = "denied"

    window_start = 0.35
    window_end   = 0.60

    affected = []
    for r in records:
        if window_start <= r["timestamp"] <= window_end:
            fv = r["feature_vector"].copy()

            # Inflate primary seasonal feature
            if primary_feat in fv and isinstance(fv[primary_feat], (int, float)):
                raw = fv[primary_feat] * 1.4
                # Clamp to value_range if available
                fv[primary_feat] = raw   # downstream clamping left to the model

            # Inflate secondary feature (drives denials)
            if secondary_feat in fv and isinstance(fv[secondary_feat], (int, float)):
                fv[secondary_feat] = min(fv[secondary_feat] + 0.15, 0.95)

            r["feature_vector"] = fv

            # Higher secondary value -> denial
            if fv.get(secondary_feat, 0) > 0.45:
                r["outcome"] = negative_outcome

            affected.append(r["id"])

    return {
        "type":        "seasonal_variation",
        "label":       "Decoy A - Seasonal Approval Rate Drop",
        "description": (
            f"Approval rates drop 23% between timestamps {window_start}-{window_end}. "
            f"Superficially resembles model drift. True cause: seasonal "
            f"demand surge with higher {primary_feat} and {secondary_feat}."
        ),
        "affected_records": affected,
        "window":           (window_start, window_end),
        "true_cause":       "seasonal_demand_surge",
        "primary_feature":  primary_feat,
        "secondary_feature": secondary_feat,
        "elimination_queries": [
            f"QUERY_FEATURE_DISTRIBUTION({primary_feat}, group_by=time_range)",
            f"QUERY_FEATURE_DISTRIBUTION({secondary_feat}, group_by=time_range)",
        ],
        "benign_explanation": (
            f"Higher {primary_feat} -> higher {secondary_feat} -> legitimate denials."
        ),
    }


def _generate_legitimate_risk_decoy(
    records: List[Dict],
    features: Dict,
    domain_context: Optional[Dict] = None,
) -> Dict:
    """
    Decoy B: A legitimate, non-protected risk feature shows high denial rate
    (looks like proxy discrimination).
    True explanation: it is a legitimate, domain-appropriate risk factor.

    Detection: QUERY_FEATURE_DISTRIBUTION reveals the high denial rate
               exists uniformly across ALL demographic groups.
    Elimination: 2 queries needed (distribution + counterfactual).
    """
    # Resolve from context or fall back to loan literals
    if domain_context:
        risk_feat  = domain_context.get("legitimate_risk_feature") or "payment_history"
        threshold  = domain_context.get("legitimate_risk_threshold")
        if threshold is None:
            threshold = 40.0
        negative_outcome = domain_context.get("negative_outcome", "denied")
    else:
        risk_feat  = "payment_history"
        threshold  = 40.0
        negative_outcome = "denied"

    affected = []
    for r in records:
        val = r["feature_vector"].get(risk_feat)
        if val is None:
            # Feature may live in proxy_vector as a fallback
            val = r.get("proxy_vector", {}).get(risk_feat)
        if isinstance(val, (int, float)) and val < threshold:
            r["outcome"] = negative_outcome
            affected.append(r["id"])

    return {
        "type":        "legitimate_risk_factor",
        "label":       f"Decoy B - {risk_feat.replace('_', ' ').title()} Denial Pattern",
        "description": (
            f"{risk_feat} < {threshold} shows 87% denial rate, creating a strong "
            f"statistical signal. Superficially resembles proxy discrimination. "
            f"True cause: {risk_feat} is a legitimate, non-protected predictor."
        ),
        "affected_records":  affected,
        "risk_feature":      risk_feat,
        "threshold":         threshold,
        "true_cause":        "legitimate_risk_signal",
        "elimination_queries": [
            f"QUERY_FEATURE_DISTRIBUTION({risk_feat}, group_by=demographic_proxy)",
            f"QUERY_COUNTERFACTUAL(rec_0010, {risk_feat}, {int(threshold * 2)})",
        ],
        "benign_explanation": (
            f"Low {risk_feat} correlates with {negative_outcome} outcomes uniformly "
            f"across all demographic groups - no protected attribute is driving this."
        ),
    }
