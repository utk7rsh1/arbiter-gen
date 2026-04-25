"""Per-component reward rubrics for ARBITER.

Each class wraps the corresponding reward.py function and returns a
RubricResult, making reward components independently observable and
ablatable. To ablate a component, remove it from
ArbiterEnvironment.get_rubrics() — reward.py stays untouched.

RubricResult is defined here because openenv-core 0.2.3 does not include it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from openenv.core.rubrics import Rubric

from .reward import (
    budget_efficiency_bonus,
    causal_chain_bonus,
    consistency_penalty,
    intermediate_claim_reward,
    terminal_reward,
)


class RubricResult(BaseModel):
    """Structured output from a single rubric evaluation."""

    name:        str
    description: str
    score:       float
    max_score:   float


# ── Rubric classes ─────────────────────────────────────────────────────────────

class IntermediateClaimRubric(Rubric):
    """Reward per causal, counterfactual, or theory-of-mind claim."""

    name        = "intermediate_claims"
    description = "Per-claim reward for causal, counterfactual, and theory-of-mind claims"
    max_score   = 30.0  # ~10 claims × max 3.0 (ToM bonus)

    def forward(self, action: Any, observation: Any) -> float:
        # Primary path is evaluate(); forward() satisfies the abstract requirement.
        return 0.0

    def evaluate(
        self,
        verification_result: Optional[Dict] = None,
        claim_total: Optional[float] = None,
    ) -> RubricResult:
        """Evaluate a single step's claim or an episode's accumulated total.

        Pass ``claim_total`` on terminal steps (pre-summed by the env) and
        ``verification_result`` on per-claim steps.
        """
        if claim_total is not None:
            score = claim_total
        elif verification_result is not None:
            score = intermediate_claim_reward(verification_result)
        else:
            score = 0.0
        return RubricResult(
            name=self.name,
            description=self.description,
            score=score,
            max_score=self.max_score,
        )


class CausalChainRubric(Rubric):
    """Bonus for correctly tracing the full causal path to the anomaly."""

    name        = "causal_chain"
    description = "Bonus for correctly tracing the full causal path to the anomaly"
    max_score   = 10.0  # REWARD_CHAIN_MULTIPLIER × max chain length ≈ 5

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def evaluate(
        self,
        claimed_chain: List[str] = (),
        true_chain: List[str] = (),
    ) -> RubricResult:
        score = causal_chain_bonus(list(claimed_chain), list(true_chain))
        return RubricResult(
            name=self.name,
            description=self.description,
            score=score,
            max_score=self.max_score,
        )


class ConsistencyRubric(Rubric):
    """Penalty for contradictory claim pairs detected by the Meta-Overseer."""

    name        = "consistency"
    description = "Penalty for contradictory claim pairs detected by the Meta-Overseer"
    max_score   = 0.0  # penalty-only rubric

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def evaluate(self, num_violations: int = 0) -> RubricResult:
        score = consistency_penalty(num_violations)
        return RubricResult(
            name=self.name,
            description=self.description,
            score=score,
            max_score=self.max_score,
        )


class BudgetEfficiencyRubric(Rubric):
    """Reward for conserving query budget at episode end."""

    name        = "budget_efficiency"
    description = "Reward for conserving query budget at episode end"
    max_score   = 6.0  # REWARD_BUDGET_EFFICIENCY × QUERY_BUDGET = 0.3 × 20

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def evaluate(self, remaining_budget: int = 0) -> RubricResult:
        score = budget_efficiency_bonus(remaining_budget)
        return RubricResult(
            name=self.name,
            description=self.description,
            score=score,
            max_score=self.max_score,
        )


class TerminalRubric(Rubric):
    """Terminal reward for correct anomaly type, demographic, action, and decoys."""

    name        = "terminal_verdict"
    description = (
        "Terminal reward for correct anomaly type, affected demographic, "
        "recommended action, and decoy elimination"
    )
    max_score   = 20.0  # 10 + 5 + 3 + 2

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def evaluate(
        self,
        verdict: Optional[Dict] = None,
        anomaly_info: Optional[Dict] = None,
        decoy_states: Optional[Dict] = None,
    ) -> RubricResult:
        if verdict is None or anomaly_info is None:
            score = 0.0
        else:
            result = terminal_reward(verdict, anomaly_info, decoy_states or {})
            score = result["terminal_total"]
        return RubricResult(
            name=self.name,
            description=self.description,
            score=score,
            max_score=self.max_score,
        )
