"""Level 6 — Schema Drift (Patronus AI sub-theme).

Mid-episode, the regulatory framework changes:
  - A previously acceptable proxy feature is reclassified as non-compliant.
  - The Auditor must DETECT the schema change and re-evaluate its causal claims.
  - Any claim made BEFORE the drift using the newly-banned feature earns a bonus
    if the Auditor catches the reclassification and flags it.

New action: FLAG_SCHEMA_CHANGE(feature_id, reason)
New reward component: +4.0 if schema change correctly identified (feature + step window).
Penalty: -2.0 if Auditor submits verdict without acknowledging a schema change.
"""
import random
from typing import Dict, Optional

# Regulatory schema: maps feature_id -> compliant/non-compliant
# At drift point, one feature flips from compliant to non_compliant
SCHEMA_VERSIONS = {
    "pre_drift": {
        "zip_code_cluster":        "compliant",
        "surname_pattern":         "compliant",
        "neighborhood_density":    "compliant",
        "credit_score":            "compliant",
        "employment_status":       "compliant",
        "debt_ratio":              "compliant",
        "income":                  "compliant",
        "loan_amount":             "compliant",
        "payment_history":         "compliant",
    },
    "post_drift": {
        # zip_code_cluster becomes non-compliant after new EU-style regulation
        "zip_code_cluster":        "non_compliant",
        "surname_pattern":         "non_compliant",
        "neighborhood_density":    "compliant",   # stays ok
        "credit_score":            "compliant",
        "employment_status":       "compliant",
        "debt_ratio":              "compliant",
        "income":                  "compliant",
        "loan_amount":             "compliant",
        "payment_history":         "compliant",
    },
}

SCHEMA_CHANGE_REWARD     = 4.0   # correctly flagging the schema change
SCHEMA_MISSED_PENALTY    = -2.0  # submitting without flagging when drift occurred
SCHEMA_WRONG_FEATURE_PEN = -1.0  # flagging the wrong feature


def get_drift_step(total_steps: int = 20, seed: Optional[int] = None) -> int:
    """Return the step at which schema drift fires (between step 6 and 14)."""
    rng = random.Random(seed)
    return rng.randint(6, 14)


def apply_schema_drift(ep_data: Dict, drift_step: int) -> Dict:
    """Attach schema drift metadata to episode data."""
    ep_data["schema_drift"] = {
        "enabled":        True,
        "drift_step":     drift_step,
        "pre_schema":     SCHEMA_VERSIONS["pre_drift"],
        "post_schema":    SCHEMA_VERSIONS["post_drift"],
        "changed_features": ["zip_code_cluster", "surname_pattern"],
        "regulation_name": "AI Fairness Directive 2026",
        "change_description": (
            "Under the new AI Fairness Directive 2026, use of geographic "
            "cluster proxies (zip_code_cluster) and surname patterns in "
            "credit decisions is now explicitly prohibited."
        ),
    }
    return ep_data


def verify_schema_change_flag(
    claim_feature: str,
    claim_step: int,
    drift_step: int,
    changed_features: list,
) -> Dict:
    """
    Verify a FLAG_SCHEMA_CHANGE action.

    Returns dict with:
        correct_feature: bool
        correct_timing: bool  (flagged within 4 steps of drift)
        reward: float
    """
    correct_feature = claim_feature in changed_features
    correct_timing  = abs(claim_step - drift_step) <= 4

    if correct_feature and correct_timing:
        reward = SCHEMA_CHANGE_REWARD
    elif correct_feature and not correct_timing:
        reward = SCHEMA_CHANGE_REWARD * 0.5   # partial credit
    else:
        reward = SCHEMA_WRONG_FEATURE_PEN

    return {
        "correct_feature": correct_feature,
        "correct_timing":  correct_timing,
        "reward":          reward,
        "drift_step":      drift_step,
        "changed_features": changed_features,
    }


def get_current_schema(step: int, drift_step: int) -> Dict:
    """Return the regulatory schema applicable at the given step."""
    if step < drift_step:
        return SCHEMA_VERSIONS["pre_drift"]
    return SCHEMA_VERSIONS["post_drift"]


def schema_drift_observation(step: int, drift_step: int) -> Optional[Dict]:
    """
    Return a schema-change notification if this is the drift step.
    Injected into the observation so the Auditor can see it.
    """
    if step == drift_step:
        return {
            "schema_change_alert": True,
            "regulation":          "AI Fairness Directive 2026",
            "newly_non_compliant": ["zip_code_cluster", "surname_pattern"],
            "effective_immediately": True,
        }
    return None
