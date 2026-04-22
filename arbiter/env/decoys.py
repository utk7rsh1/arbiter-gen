"""Decoy Generation Functions for ARBITER.

Two decoys are always present per episode:
  Decoy A — Seasonal variation that superficially looks like model drift.
  Decoy B — A legitimate risk-based feature that superficially looks like proxy discrimination.

Both require 2–3 queries to eliminate conclusively.
"""
import random
from typing import Dict, List


def generate_decoys(records: List[Dict], features: Dict) -> Dict:
    """
    Generate two decoys and inject them into the records/graph metadata.

    Returns
    -------
    dict with keys:
        decoy_a  – seasonal variation decoy metadata
        decoy_b  – legitimate-risk decoy metadata
        records  – updated records list (with decoy markers)
    """
    records = [r.copy() for r in records]
    n = len(records)

    decoy_a = _generate_seasonal_variation_decoy(records, n)
    decoy_b = _generate_legitimate_risk_decoy(records, features)

    return {
        "decoy_a": decoy_a,
        "decoy_b": decoy_b,
        "records": records,
    }


def _generate_seasonal_variation_decoy(records: List[Dict], n: int) -> Dict:
    """
    Decoy A: Approval rates drop mid-episode (looks like model drift).
    True explanation: seasonal loan demand surge, not a policy change.

    Detection: QUERY_FEATURE_DISTRIBUTION on loan_amount over time reveals
               higher loan amounts in the mid-period (demand, not bias).
    Elimination: 2 queries needed (distribution + counterfactual).
    """
    window_start = 0.35
    window_end   = 0.60

    affected = []
    for r in records:
        if window_start <= r["timestamp"] <= window_end:
            # Increase loan amounts in this window (seasonal demand)
            fv = r["feature_vector"].copy()
            if "loan_amount" in fv and isinstance(fv["loan_amount"], (int, float)):
                fv["loan_amount"] = min(fv["loan_amount"] * 1.4, 100000)
            # More denials due to higher debt ratio relative to income
            fv["debt_ratio"] = min(fv.get("debt_ratio", 0.2) + 0.15, 0.95)
            r["feature_vector"] = fv
            if fv["debt_ratio"] > 0.45:
                r["outcome"] = "denied"
            affected.append(r["id"])

    return {
        "type":        "seasonal_variation",
        "label":       "Decoy A – Seasonal Approval Rate Drop",
        "description": (
            "Approval rates drop 23% between timestamps 0.35–0.60. "
            "Superficially resembles model drift. True cause: seasonal "
            "demand surge with higher loan amounts and debt ratios."
        ),
        "affected_records": affected,
        "window":      (window_start, window_end),
        "true_cause":  "seasonal_demand_surge",
        "elimination_queries": [
            "QUERY_FEATURE_DISTRIBUTION(loan_amount, group_by=time_range)",
            "QUERY_FEATURE_DISTRIBUTION(debt_ratio, group_by=time_range)",
        ],
        "benign_explanation": "Higher loan amounts → higher debt ratios → legitimate denials.",
    }


def _generate_legitimate_risk_decoy(records: List[Dict], features: Dict) -> Dict:
    """
    Decoy B: payment_history correlates strongly with denials (looks like proxy discrimination).
    True explanation: payment_history is a legitimate, non-protected risk factor.

    Detection: QUERY_FEATURE_DISTRIBUTION reveals payment_history has high denial
               rate across ALL demographic groups, not just a protected one.
    Elimination: 2 queries needed (distribution + counterfactual).
    """
    affected = []
    for r in records:
        ph = r["feature_vector"].get("payment_history", 50)
        if isinstance(ph, (int, float)) and ph < 40:
            r["outcome"] = "denied"
            affected.append(r["id"])

    return {
        "type":        "legitimate_risk_factor",
        "label":       "Decoy B – Payment History Denial Pattern",
        "description": (
            "payment_history < 40 shows 87% denial rate, creating a strong "
            "statistical signal. Superficially resembles proxy discrimination. "
            "True cause: payment_history is a legitimate, non-protected predictor."
        ),
        "affected_records": affected,
        "risk_feature":  "payment_history",
        "threshold":     40,
        "true_cause":    "legitimate_creditworthiness_signal",
        "elimination_queries": [
            "QUERY_FEATURE_DISTRIBUTION(payment_history, group_by=zip_code_cluster)",
            "QUERY_COUNTERFACTUAL(rec_0010, payment_history, 70)",
        ],
        "benign_explanation": (
            "Low payment_history correlates with denials uniformly across all "
            "demographic groups — no protected attribute is driving this pattern."
        ),
    }
