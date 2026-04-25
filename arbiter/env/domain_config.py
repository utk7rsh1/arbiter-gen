"""Domain Configuration Schema for ARBITER generalisation.

Defines DomainConfig and FeatureSpec — the single source of truth between
the Groq generator and every downstream pipeline module.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class FeatureSpec(BaseModel):
    """Specification for a single feature in a decision system."""

    name: str = Field(description="Snake_case feature identifier, e.g. 'technical_score'")
    description: str = Field(description="Human-readable description of the feature")
    dtype: str = Field(description="One of: 'continuous', 'categorical', 'binary'")
    value_range: Optional[Tuple[float, float]] = Field(
        default=None, description="(min, max) for continuous features"
    )
    categories: Optional[List[str]] = Field(
        default=None, description="Allowed values for categorical features"
    )
    protected_attribute: Optional[str] = Field(
        default=None,
        description="For proxy features: which protected attribute this correlates with "
                    "(e.g. 'race', 'gender', 'age', 'religion', 'nationality', "
                    "'disability', 'socioeconomic_status')",
    )

    def sample_values(self, steps: int = 20) -> list:
        """Generate a discrete list of values suitable for random.choice()."""
        if self.dtype == "continuous" and self.value_range:
            lo, hi = self.value_range
            return [round(lo + i * (hi - lo) / steps, 4) for i in range(steps + 1)]
        elif self.dtype == "categorical" and self.categories:
            return list(self.categories)
        else:  # binary or fallback
            return [0, 1]

    def to_graph_dict(self, is_proxy: bool = False, is_hidden: bool = False) -> dict:
        """Convert to the internal graph feature dict used by graph.py."""
        d = {
            "id":        self.name,
            "name":      self.name.replace("_", " ").title(),
            "proxy":     is_proxy,
            "hidden":    is_hidden,
            "protected": is_proxy,
            "values":    self.sample_values(),
        }
        if is_proxy and self.protected_attribute:
            d["correlates"] = self.protected_attribute
        return d


class DomainConfig(BaseModel):
    """Complete configuration for one AI decision domain.

    This is the contract between GroqGraphGenerator and every downstream
    module (graph.py, environment.py, sft_generator.py, etc.).
    Never hardcode domain constants in those modules — read from this object.
    """

    domain_name: str = Field(
        description="Human-readable name, e.g. 'Software Engineering Hiring'"
    )
    decision_verb: str = Field(
        description="Past-tense verb for the decision, e.g. 'hired' / 'rejected'"
    )
    positive_outcome: str = Field(
        description="Label for the favourable outcome, e.g. 'hired'"
    )
    negative_outcome: str = Field(
        description="Label for the unfavourable outcome, e.g. 'rejected'"
    )

    # Feature catalogues
    explicit_features: List[FeatureSpec] = Field(
        description="Exactly 6 observable, legitimate inputs to the decision"
    )
    proxy_features: List[FeatureSpec] = Field(
        description="Exactly 3 features that correlate with protected attributes"
    )
    hidden_features: List[FeatureSpec] = Field(
        description="Exactly 2 internal scores the system uses but does not expose"
    )
    outcome_nodes: List[str] = Field(
        description="Names of intermediate decision outcome nodes"
    )

    # Threshold
    approval_threshold_description: str = Field(
        description="Human-readable threshold rule, e.g. 'technical_score > 70'"
    )
    approval_threshold_feature: str = Field(
        description="Name of the feature used in the threshold comparison"
    )
    approval_threshold_value: float = Field(
        description="Numeric cutoff value"
    )

    # Discrimination target
    discriminated_group_value: str = Field(
        description="The category value that is discriminated against, "
                    "e.g. 'cluster_7' or 'low_income_zip'"
    )
    discriminated_group_feature: str = Field(
        description="Name of the proxy feature that carries the discrimination; "
                    "must be one of the proxy_features names"
    )

    # Narrative
    anomaly_description: str = Field(
        description="Plain-English explanation of the bias"
    )
    causal_chain: List[str] = Field(
        description="Ground truth: [proxy_feature_name, hidden_feature_name, outcome_node_name]"
    )
    system_description: str = Field(
        description="2-3 sentence description of the AI system for the auditor system prompt"
    )

    # ── v2 fields: decoy and schema-drift configuration ───────────────────────
    # All Optional so existing cached v1 configs still validate.

    seasonal_decoy_feature: Optional[str] = Field(
        default=None,
        description="Name of a continuous explicit feature whose values appear "
                    "seasonally inflated mid-episode (e.g. 'loan_amount', 'claim_amount'). "
                    "Must be one of explicit_features names.",
    )
    seasonal_decoy_secondary_feature: Optional[str] = Field(
        default=None,
        description="Name of a second continuous explicit feature that rises with "
                    "the seasonal effect, driving more denials (e.g. 'debt_ratio'). "
                    "Must be one of explicit_features names.",
    )
    legitimate_risk_feature: Optional[str] = Field(
        default=None,
        description="Name of a non-protected continuous explicit feature that is a "
                    "legitimate denial driver (Decoy B). "
                    "Must be one of explicit_features names.",
    )
    legitimate_risk_threshold: Optional[float] = Field(
        default=None,
        description="Records whose legitimate_risk_feature value is BELOW this "
                    "cutoff are forced to the negative outcome as Decoy B.",
    )
    drift_regulation_name: Optional[str] = Field(
        default=None,
        description="Name of the regulatory directive that triggers schema drift, "
                    "e.g. 'AI Fairness Directive 2026' or 'EEOC Disparate Impact Rule'.",
    )
    newly_non_compliant_features: Optional[List[str]] = Field(
        default=None,
        description="Subset of proxy_features names that become non-compliant after "
                    "schema drift. Must include discriminated_group_feature.",
    )
    drift_change_description: Optional[str] = Field(
        default=None,
        description="One-sentence narrative shown in the schema-change alert, "
                    "e.g. 'Under the new regulation, use of X in Y decisions is prohibited.'",
    )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def all_feature_names(self) -> List[str]:
        return (
            [f.name for f in self.explicit_features]
            + [f.name for f in self.proxy_features]
            + [f.name for f in self.hidden_features]
        )

    def denial_outcome_id(self) -> str:
        return f"{self.negative_outcome}_rate_overall"

    def approval_outcome_id(self) -> str:
        return f"{self.positive_outcome}_rate_overall"

    def minority_outcome_id(self) -> str:
        return f"{self.negative_outcome}_rate_minority"

    def build_outcome_nodes(self) -> list:
        """Return a list of outcome-node dicts in the graph.py internal format."""
        neg = self.negative_outcome
        pos = self.positive_outcome
        return [
            {"id": f"{neg}_rate_overall",    "name": f"Overall {neg.title()} Rate"},
            {"id": f"{pos}_rate_overall",    "name": f"Overall {pos.title()} Rate"},
            {"id": "flag_rate_overall",      "name": "Overall Flag Rate"},
            {"id": f"{neg}_rate_minority",   "name": f"{neg.title()} Rate - Minority Demographic"},
            {"id": f"{pos}_rate_majority",   "name": f"{pos.title()} Rate - Majority Demographic"},
        ]

    def resolve_defaults(self) -> "DomainConfig":
        """Fill any missing v2 fields using data already in the config.

        This is called after Groq parsing to normalise configs that predate
        schema v2 or where the LLM omitted the optional fields.

        Returns self (mutates in-place for convenience, also returns for chaining).
        """
        explicit_names = [f.name for f in self.explicit_features]
        continuous_explicit = [
            f for f in self.explicit_features if f.dtype == "continuous"
        ]

        # seasonal_decoy_feature: first continuous explicit feature
        if self.seasonal_decoy_feature is None and continuous_explicit:
            object.__setattr__(self, "seasonal_decoy_feature", continuous_explicit[0].name)

        # seasonal_decoy_secondary_feature: any continuous explicit feature != primary.
        # Scan the full list so domains with only 1 continuous feature don't silently
        # assign primary == secondary (which makes Decoy A double-inflate one feature).
        if self.seasonal_decoy_secondary_feature is None:
            candidate = next(
                (f for f in continuous_explicit if f.name != self.seasonal_decoy_feature),
                None,
            )
            # Only fall back to the primary feature when there is genuinely no alternative.
            fallback = continuous_explicit[0] if continuous_explicit else None
            secondary = candidate if candidate is not None else fallback
            if secondary is not None:
                object.__setattr__(self, "seasonal_decoy_secondary_feature", secondary.name)

        # legitimate_risk_feature: third continuous explicit (fallback to second, then first)
        if self.legitimate_risk_feature is None and continuous_explicit:
            for feat in continuous_explicit:
                if feat.name not in (self.seasonal_decoy_feature, self.seasonal_decoy_secondary_feature):
                    object.__setattr__(self, "legitimate_risk_feature", feat.name)
                    break
            else:
                object.__setattr__(self, "legitimate_risk_feature", continuous_explicit[0].name)

        # legitimate_risk_threshold: 30% into the value range, or 40 for 0-100 ranges
        if self.legitimate_risk_threshold is None and self.legitimate_risk_feature:
            feat_spec = next(
                (f for f in self.explicit_features if f.name == self.legitimate_risk_feature),
                None,
            )
            if feat_spec and feat_spec.value_range:
                lo, hi = feat_spec.value_range
                threshold = round(lo + 0.30 * (hi - lo), 2)
            else:
                threshold = 40.0
            object.__setattr__(self, "legitimate_risk_threshold", threshold)

        # drift_regulation_name: generic fallback
        if self.drift_regulation_name is None:
            object.__setattr__(self, "drift_regulation_name", "AI Fairness Directive 2026")

        # newly_non_compliant_features: at minimum the discriminated feature
        if self.newly_non_compliant_features is None:
            object.__setattr__(
                self, "newly_non_compliant_features", [self.discriminated_group_feature]
            )

        # drift_change_description: templated fallback
        if self.drift_change_description is None:
            features_str = ", ".join(self.newly_non_compliant_features or [])
            desc = (
                f"Under the {self.drift_regulation_name}, use of "
                f"{features_str} in {self.domain_name.lower()} decisions "
                f"is now explicitly prohibited."
            )
            object.__setattr__(self, "drift_change_description", desc)

        return self
