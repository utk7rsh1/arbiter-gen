"""Claim Schemas and Verification Engine for ARBITER.

Three claim types:
  CausalLinkClaim        – asserts a cause→effect relationship
  CounterfactualClaim    – predicts outcome under feature change (pays 2× reward)
  TheoryOfMindClaim      – asserts Defender is actively hiding something (Level 4+)

Verification returns per-field correctness and a total score.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Claim Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class CausalLinkClaim:
    cause_feature:  str
    effect_outcome: str
    mechanism:      str                    # mediating node
    direction:      str                    # "positive" | "negative"
    confidence:     str                    # "HIGH" | "MEDIUM" | "LOW"
    basis_records:  List[str]
    anomaly_type:   str                    # "proxy_discrimination" | "adversarial_injection" | "model_drift"
    step:           int = 0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class CounterfactualClaim:
    subject_record:           str
    counterfactual_feature:   str
    predicted_outcome_change: str          # "approved" | "denied" | "flagged" | "no_change"
    confidence:               str
    basis:                    str
    step:                     int = 0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class TheoryOfMindClaim:
    defender_action:    str                # "obfuscating" | "injecting" | "manipulating"
    target_link:        str                # e.g. "zip_code_cluster_7→internal_risk_score"
    obfuscation_method: str                # "link_substitution" | "record_injection" | "proxy_laundering" | "timestamp_manipulation"
    confidence:         str
    basis:              str
    step:               int = 0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


# ── Verification ──────────────────────────────────────────────────────────────

def verify_causal_claim(claim: CausalLinkClaim, anomaly_info: Dict) -> Dict:
    """
    Verify a CausalLinkClaim against ground truth.

    Returns
    -------
    dict:
        field_scores  – {field: bool}
        correct_count – int
        total_fields  – int
        score         – float [0, 1.0]
        high_conf_penalty – float (0 or -0.5)
    """
    chain   = anomaly_info.get("causal_chain", [])
    atype   = anomaly_info.get("type", 0)
    atype_map = {1: "proxy_discrimination", 2: "adversarial_injection", 3: "model_drift"}
    correct_atype = atype_map.get(atype, "unknown")

    proxy_feat  = anomaly_info.get("proxy_feature") or anomaly_info.get("injected_feature") or anomaly_info.get("post_drift_cause", "")
    hidden_med  = anomaly_info.get("hidden_mediator") or anomaly_info.get("drift_mediator", "")
    outcome     = anomaly_info.get("outcome", "denial_rate_overall")

    scores = {
        "cause_feature":  claim.cause_feature  in chain or claim.cause_feature == proxy_feat,
        "effect_outcome": claim.effect_outcome in chain or claim.effect_outcome == outcome,
        "mechanism":      claim.mechanism == hidden_med or (len(chain) <= 2 and claim.mechanism in chain),
        "direction":      claim.direction == "positive",
        "anomaly_type":   claim.anomaly_type == correct_atype,
    }

    correct = sum(scores.values())
    total   = len(scores)
    base_score = correct / total

    high_conf_penalty = 0.0
    if claim.confidence == "HIGH" and (total - correct) > 1:
        high_conf_penalty = -0.5

    return {
        "field_scores":       scores,
        "correct_count":      correct,
        "total_fields":       total,
        "score":              round(base_score, 3),
        "high_conf_penalty":  high_conf_penalty,
        "claim_type":         "causal",
    }


def verify_counterfactual_claim(claim: CounterfactualClaim, cf_result: Dict) -> Dict:
    """
    Verify a CounterfactualClaim against the computed counterfactual result.
    Pays 2× reward compared to causal claims.
    """
    computed_outcome = cf_result.get("counterfactual_outcome", "denied")
    changed          = cf_result.get("changed", False)

    predicted = claim.predicted_outcome_change
    if predicted == "no_change":
        correct = not changed
    else:
        correct = predicted == computed_outcome

    score = 1.0 if correct else 0.0

    high_conf_penalty = 0.0
    if claim.confidence == "HIGH" and not correct:
        high_conf_penalty = -0.5

    return {
        "field_scores":      {"predicted_outcome_change": correct},
        "correct_count":     int(correct),
        "total_fields":      1,
        "score":             score,
        "reward":            score * 2.0,   # counterfactual pays double
        "high_conf_penalty": high_conf_penalty,
        "claim_type":        "counterfactual",
    }


def verify_theory_of_mind_claim(claim: TheoryOfMindClaim, defender_action_log: List[Dict]) -> Dict:
    """
    Verify a TheoryOfMindClaim against the Defender's actual action log.
    Earns +3.0 bonus if fully correct.
    """
    if not defender_action_log:
        return {"correct": False, "score": 0.0, "bonus": 0.0, "claim_type": "theory_of_mind"}

    last_action = defender_action_log[-1]
    correct_method = last_action.get("method", "")
    correct_target = last_action.get("target_link", "")

    method_correct = claim.obfuscation_method == correct_method
    target_correct = claim.target_link == correct_target or claim.target_link in correct_target
    action_correct = claim.defender_action == "obfuscating"

    all_correct = method_correct and target_correct and action_correct

    return {
        "field_scores": {
            "defender_action":    action_correct,
            "target_link":        target_correct,
            "obfuscation_method": method_correct,
        },
        "correct":    all_correct,
        "score":      1.0 if all_correct else (0.5 if (method_correct or target_correct) else 0.0),
        "bonus":      3.0 if all_correct else 0.0,
        "claim_type": "theory_of_mind",
    }
