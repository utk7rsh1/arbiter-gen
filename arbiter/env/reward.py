"""Reward Function for ARBITER — all 8 components.

Components:
  1  Intermediate claim reward       [0, 1.0] per causal claim, [0, 2.0] per counterfactual
  2  High-confidence penalty         -0.5 if HIGH-conf claim has >1 wrong field
  3  Theory-of-mind bonus            +3.0 if fully correct
  4  Causal chain completeness       +2.0 × chain_length for full path trace
  5  Meta-Overseer consistency       -1.0 per contradictory claim pair
  6  Budget efficiency               +0.3 × remaining_budget at episode end
  7  Terminal verdict rewards        +10/+5/+3/+2 for correct type/demographic/action/decoys
  8  Aggregation                     compute_episode_reward() sums everything
"""
from typing import Dict, List, Any

from config import (
    REWARD_CAUSAL_MAX, REWARD_COUNTERFACTUAL_MAX, REWARD_HIGH_CONF_PENALTY,
    REWARD_TOM_BONUS, REWARD_CHAIN_MULTIPLIER, REWARD_CONSISTENCY_PENALTY,
    REWARD_BUDGET_EFFICIENCY, REWARD_CORRECT_TYPE, REWARD_CORRECT_DEMOGRAPHIC,
    REWARD_CORRECT_ACTION, REWARD_DECOYS_ELIMINATED,
)


def intermediate_claim_reward(verification_result: Dict) -> float:
    """Reward for a single intermediate claim (causal or counterfactual)."""
    claim_type = verification_result.get("claim_type", "causal")
    score      = verification_result.get("score", 0.0)
    penalty    = verification_result.get("high_conf_penalty", 0.0)

    if claim_type == "counterfactual":
        base = score * REWARD_COUNTERFACTUAL_MAX
    elif claim_type == "theory_of_mind":
        base = verification_result.get("bonus", 0.0)
    else:
        base = score * REWARD_CAUSAL_MAX

    return round(base + penalty, 4)


def causal_chain_bonus(claimed_chain: List[str], true_chain: List[str]) -> float:
    """
    +2.0 × chain_length if the Auditor has correctly traced the full causal path.
    Partial credit: proportional to the fraction of the chain covered.
    """
    if not true_chain or not claimed_chain:
        return 0.0

    covered = sum(1 for node in true_chain if node in claimed_chain)
    fraction = covered / len(true_chain)

    if fraction >= 1.0:
        return REWARD_CHAIN_MULTIPLIER * len(true_chain)
    elif fraction >= 0.5:
        return REWARD_CHAIN_MULTIPLIER * len(true_chain) * fraction * 0.5
    return 0.0


def consistency_penalty(num_violations: int) -> float:
    """Meta-Overseer: -1.0 per contradictory claim pair."""
    return REWARD_CONSISTENCY_PENALTY * num_violations


def budget_efficiency_bonus(remaining_budget: int) -> float:
    """Reward for conserving query budget."""
    return REWARD_BUDGET_EFFICIENCY * max(0, remaining_budget)


def terminal_reward(
    verdict: Dict,
    anomaly_info: Dict,
    decoy_states: Dict,
) -> Dict:
    """
    Compute the full terminal reward on SUBMIT_REPORT.

    Parameters
    ----------
    verdict      : Auditor's submitted report dict.
    anomaly_info : Ground-truth anomaly metadata.
    decoy_states : {"decoy_a": "ELIMINATED"|"ACTIVE", "decoy_b": "ELIMINATED"|"ACTIVE"}

    Returns
    -------
    dict with individual components and total.
    """
    atype_map = {1: "proxy_discrimination", 2: "adversarial_injection", 3: "model_drift"}
    correct_type       = atype_map.get(anomaly_info.get("type", 0), "")
    correct_demographic = anomaly_info.get("affected_demographic", "")
    correct_action     = anomaly_info.get("recommended_action", "")

    r_type  = REWARD_CORRECT_TYPE       if verdict.get("anomaly_type")         == correct_type       else 0.0
    r_demo  = REWARD_CORRECT_DEMOGRAPHIC if verdict.get("affected_demographic") == correct_demographic else 0.0
    r_act   = REWARD_CORRECT_ACTION     if verdict.get("recommended_action")    == correct_action     else 0.0
    r_decoy = REWARD_DECOYS_ELIMINATED  if (
        decoy_states.get("decoy_a") == "ELIMINATED" and
        decoy_states.get("decoy_b") == "ELIMINATED"
    ) else 0.0

    total = r_type + r_demo + r_act + r_decoy
    return {
        "correct_type_reward":        r_type,
        "correct_demographic_reward": r_demo,
        "correct_action_reward":      r_act,
        "decoys_eliminated_reward":   r_decoy,
        "terminal_total":             total,
        "verdict_correct":            r_type > 0,
    }


def compute_episode_reward(
    claim_rewards:      List[float],
    consistency_violations: int,
    remaining_budget:   int,
    claimed_chain:      List[str],
    true_chain:         List[str],
    verdict:            Dict,
    anomaly_info:       Dict,
    decoy_states:       Dict,
) -> Dict:
    """Aggregate all reward components for a complete episode."""
    r_claims      = sum(claim_rewards)
    r_chain       = causal_chain_bonus(claimed_chain, true_chain)
    r_consistency = consistency_penalty(consistency_violations)
    r_budget      = budget_efficiency_bonus(remaining_budget)
    r_terminal    = terminal_reward(verdict, anomaly_info, decoy_states)

    total = r_claims + r_chain + r_consistency + r_budget + r_terminal["terminal_total"]

    return {
        "claim_reward":           round(r_claims, 3),
        "chain_bonus":            round(r_chain, 3),
        "consistency_penalty":    round(r_consistency, 3),
        "budget_bonus":           round(r_budget, 3),
        "terminal":               r_terminal,
        "total":                  round(total, 3),
        "breakdown": {
            "intermediate_claims":   round(r_claims, 3),
            "causal_chain_bonus":    round(r_chain, 3),
            "meta_overseer_penalty": round(r_consistency, 3),
            "budget_efficiency":     round(r_budget, 3),
            **{k: v for k, v in r_terminal.items()},
        },
    }
