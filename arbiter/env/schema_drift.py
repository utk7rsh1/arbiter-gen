"""Level 6 - Schema Drift (Patronus AI sub-theme).

Mid-episode, the regulatory framework changes:
  - A previously acceptable proxy feature is reclassified as non-compliant.
  - The Auditor must DETECT the schema change and re-evaluate its causal claims.
  - Any claim made BEFORE the drift using the newly-banned feature earns a bonus
    if the Auditor catches the reclassification and flags it.

New action: FLAG_SCHEMA_CHANGE(feature_id, reason)
New reward component: +4.0 if schema change correctly identified (feature + step window).
Penalty: -2.0 if Auditor submits verdict without acknowledging a schema change.

Domain-specific regulation name and newly non-compliant features are read from
the optional `domain_context` dict when passed to apply_schema_drift().
Loan-domain literals are used as a fallback so the existing loan path is unchanged.
"""
import random
from typing import Dict, List, Optional

# ── Loan-domain static fallback ───────────────────────────────────────────────
# Used only when domain_context is None (backwards-compatible loan path).

_LOAN_PRE_DRIFT = {
    "zip_code_cluster":        "compliant",
    "surname_pattern":         "compliant",
    "neighborhood_density":    "compliant",
    "credit_score":            "compliant",
    "employment_status":       "compliant",
    "debt_ratio":              "compliant",
    "income":                  "compliant",
    "loan_amount":             "compliant",
    "payment_history":         "compliant",
}

_LOAN_POST_DRIFT = {
    "zip_code_cluster":        "non_compliant",
    "surname_pattern":         "non_compliant",
    "neighborhood_density":    "compliant",
    "credit_score":            "compliant",
    "employment_status":       "compliant",
    "debt_ratio":              "compliant",
    "income":                  "compliant",
    "loan_amount":             "compliant",
    "payment_history":         "compliant",
}

_LOAN_CHANGED_FEATURES   = ["zip_code_cluster", "surname_pattern"]
_LOAN_REGULATION_NAME    = "AI Fairness Directive 2026"
_LOAN_CHANGE_DESCRIPTION = (
    "Under the new AI Fairness Directive 2026, use of geographic "
    "cluster proxies (zip_code_cluster) and surname patterns in "
    "credit decisions is now explicitly prohibited."
)

SCHEMA_CHANGE_REWARD     = 4.0   # correctly flagging the schema change
SCHEMA_MISSED_PENALTY    = -2.0  # submitting without flagging when drift occurred
SCHEMA_WRONG_FEATURE_PEN = -1.0  # flagging the wrong feature


# ── Schema builder ────────────────────────────────────────────────────────────

def build_schema_versions(
    domain_context: Optional[Dict],
    all_feature_names: Optional[List[str]] = None,
) -> Dict:
    """Build pre-drift and post-drift schema dicts from domain_context.

    Parameters
    ----------
    domain_context   : G.graph["domain_context"] dict or None.
    all_feature_names: Full list of feature ids (explicit+proxy+hidden).
                       Used when domain_context is None to build the loan schema.

    Returns
    -------
    dict with keys "pre_drift" and "post_drift", each mapping feature_id -> status.
    """
    if domain_context is None:
        return {"pre_drift": _LOAN_PRE_DRIFT, "post_drift": _LOAN_POST_DRIFT}

    newly_non_compliant = domain_context.get("newly_non_compliant_features") or []
    feature_names = all_feature_names or []

    pre_drift  = {f: "compliant" for f in feature_names}
    post_drift = {
        f: ("non_compliant" if f in newly_non_compliant else "compliant")
        for f in feature_names
    }

    return {"pre_drift": pre_drift, "post_drift": post_drift}


# ── Public API ────────────────────────────────────────────────────────────────

def get_drift_step(seed: Optional[int] = None) -> int:
    """Return the step at which schema drift fires (between step 6 and 14)."""
    rng = random.Random(seed)
    return rng.randint(6, 14)


def apply_schema_drift(
    ep_data: Dict,
    drift_step: int,
    domain_context: Optional[Dict] = None,
    all_feature_names: Optional[List[str]] = None,
) -> Dict:
    """Attach schema drift metadata to episode data.

    Parameters
    ----------
    ep_data          : Episode data dict (mutated in-place and returned).
    drift_step       : Step at which the regulatory schema changes.
    domain_context   : G.graph["domain_context"] dict or None (-> loan fallback).
    all_feature_names: Full feature id list for building schema dicts.
    """
    schema_versions = build_schema_versions(domain_context, all_feature_names)

    if domain_context:
        regulation_name    = domain_context.get("drift_regulation_name") or _LOAN_REGULATION_NAME
        changed_features   = list(domain_context.get("newly_non_compliant_features") or [])
        change_description = domain_context.get("drift_change_description") or (
            f"Under the {regulation_name}, use of "
            f"{', '.join(changed_features)} in decisions is now explicitly prohibited."
        )
    else:
        regulation_name    = _LOAN_REGULATION_NAME
        changed_features   = _LOAN_CHANGED_FEATURES
        change_description = _LOAN_CHANGE_DESCRIPTION

    ep_data["schema_drift"] = {
        "enabled":            True,
        "drift_step":         drift_step,
        "pre_schema":         schema_versions["pre_drift"],
        "post_schema":        schema_versions["post_drift"],
        "changed_features":   changed_features,
        "regulation_name":    regulation_name,
        "change_description": change_description,
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


def get_current_schema(
    step: int,
    drift_step: int,
    domain_context: Optional[Dict] = None,
    all_feature_names: Optional[List[str]] = None,
) -> Dict:
    """Return the regulatory schema applicable at the given step."""
    schema_versions = build_schema_versions(domain_context, all_feature_names)
    if step < drift_step:
        return schema_versions["pre_drift"]
    return schema_versions["post_drift"]


def schema_drift_observation(
    step: int,
    drift_step: int,
    domain_context: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Return a schema-change notification if this is the drift step.
    Injected into the observation so the Auditor can see it.
    """
    if step == drift_step:
        if domain_context:
            regulation   = domain_context.get("drift_regulation_name") or _LOAN_REGULATION_NAME
            newly_non_compliant = list(
                domain_context.get("newly_non_compliant_features") or []
            )
        else:
            regulation          = _LOAN_REGULATION_NAME
            newly_non_compliant = list(_LOAN_CHANGED_FEATURES)

        return {
            "regulation":          regulation,
            "newly_non_compliant": newly_non_compliant,
            "effective_immediately": True,
        }
    return None
