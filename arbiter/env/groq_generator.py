"""Groq-powered domain config generator for ARBITER.

Takes a plain-English description of an AI decision system and returns a
validated DomainConfig ready to pass directly to generate_graph().

Usage:
    gen = GroqGraphGenerator()
    config = gen.generate_cached("A hiring AI that screens engineering resumes")
    ep_data = generate_graph(seed=42, domain=config)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from typing import Optional

from .domain_config import DomainConfig

# Bump this when the schema changes to orphan old cache files harmlessly.
_SCHEMA_VERSION = "v2"

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI ethics researcher designing synthetic test environments \
for AI bias auditing. Generate a realistic domain configuration for an AI \
decision system.

Output ONLY valid JSON matching this exact schema:
{schema}

Strict rules:
- explicit_features: exactly 6, legitimate/defensible inputs to the decision
- proxy_features: exactly 3, each correlating with a protected attribute \
(race, gender, age, religion, nationality, disability, socioeconomic_status)
- hidden_features: exactly 2, internal scores the system uses but does not expose
- All feature names must be snake_case, no spaces
- value_range for continuous features must be realistic for the domain \
(e.g. [0.0, 100.0] for a score, [20000, 200000] for salary)
- categories must be a list of short strings (no spaces inside values)
- discriminated_group_value must be a valid entry from the categories list \
of the discriminated_group_feature
- discriminated_group_feature must be exactly one of the proxy_features names
- approval_threshold_feature must be exactly one of the explicit_features names
- causal_chain must be exactly [proxy_feature_name, hidden_feature_name, outcome_node_name]
  where proxy_feature_name is one of proxy_features names,
  hidden_feature_name is one of hidden_features names,
  and outcome_node_name is one of outcome_nodes entries
- outcome_nodes should contain at least one entry that ends in \
'_rate_overall' pattern and can serve as the causal chain terminus
- positive_outcome and negative_outcome must be single lowercase words \
(e.g. "approved"/"denied", "hired"/"rejected", "admitted"/"rejected")
- seasonal_decoy_feature must be the name of one of the explicit_features; \
pick the most naturally "seasonal" continuous one (e.g. claim_amount, loan_amount)
- seasonal_decoy_secondary_feature must also be a name from explicit_features; \
pick a second continuous feature that would rise together with the seasonal effect
- legitimate_risk_feature must be a name from explicit_features; \
pick a continuous feature that is a legitimate, non-protected denial driver
- legitimate_risk_threshold must be a number that lies inside \
legitimate_risk_feature's value_range; records below this threshold are legitimately risky
- newly_non_compliant_features must be a non-empty list of proxy_features names \
and MUST include discriminated_group_feature
- drift_regulation_name is the name of a plausible regulatory directive \
(e.g. "AI Fairness Directive 2026")
- drift_change_description is one sentence explaining the new prohibition
"""

# ── Cache ─────────────────────────────────────────────────────────────────────

CACHE_DIR = pathlib.Path("groq_cache")


# ── Generator ─────────────────────────────────────────────────────────────────

class GroqGraphGenerator:
    """Generate DomainConfig objects from plain-English domain descriptions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package is required. Install it with: pip install groq"
            )
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, domain_description: str, seed: int = 42) -> DomainConfig:
        """Generate and validate a DomainConfig for the given domain description.

        Args:
            domain_description: Plain-English description, e.g.
                "A hiring AI that screens software engineering resumes"
            seed: Groq seed for reproducibility.

        Returns:
            A validated DomainConfig with all v2 fields resolved via
            resolve_defaults().

        Raises:
            ValueError: If Groq returns invalid JSON or schema-failing data.
        """
        schema_str = json.dumps(DomainConfig.model_json_schema(), indent=2)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(schema=schema_str),
                },
                {
                    "role": "user",
                    "content": f"Generate a domain config for: {domain_description}",
                },
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
            seed=seed,
        )

        raw = response.choices[0].message.content
        data = self._parse_json(raw)
        config = self._build_config(data)
        self._validate_consistency(config)
        config.resolve_defaults()
        return config

    def generate_cached(
        self, domain_description: str, seed: int = 42
    ) -> DomainConfig:
        """generate() with on-disk JSON caching keyed by (description, seed, schema_version).

        Repeated calls with the same arguments skip the Groq API entirely.
        Cache files are stored in ./groq_cache/<hash>.json.
        Old v1 cache files are orphaned (safe to delete manually).
        """
        cache_key = hashlib.md5(
            f"{domain_description.strip()}{seed}{_SCHEMA_VERSION}".encode()
        ).hexdigest()[:12]
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if cache_file.exists():
            try:
                config = DomainConfig(**json.loads(cache_file.read_text()))
                config.resolve_defaults()
                return config
            except Exception:
                # Stale or corrupt cache — regenerate
                cache_file.unlink(missing_ok=True)

        config = self.generate(domain_description, seed)
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file.write_text(config.model_dump_json(indent=2))
        return config

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from raw Groq response (handles stray markdown fences)."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Groq returned invalid JSON: {exc}\nRaw:\n{raw[:500]}"
            )

    def _build_config(self, data: dict) -> DomainConfig:
        """Parse raw dict into a DomainConfig, applying light normalisations."""
        # Groq sometimes omits optional fields — ensure they exist with defaults
        for feat_list_key in ("explicit_features", "proxy_features", "hidden_features"):
            for feat in data.get(feat_list_key, []):
                feat.setdefault("description", feat.get("name", ""))
                if feat.get("dtype") == "continuous":
                    feat.setdefault("value_range", None)
                if feat.get("dtype") == "categorical":
                    feat.setdefault("categories", None)

        # Sanitise category values: strip spaces (graph.py uses == comparison)
        for feat_list_key in ("proxy_features",):
            for feat in data.get(feat_list_key, []):
                if feat.get("categories"):
                    feat["categories"] = [
                        c.replace(" ", "_") for c in feat["categories"]
                    ]

        # Sanitise discriminated_group_value to match categories
        dgv = data.get("discriminated_group_value", "")
        if dgv:
            data["discriminated_group_value"] = dgv.replace(" ", "_")

        try:
            return DomainConfig(**data)
        except Exception as exc:
            raise ValueError(
                f"Groq JSON does not match DomainConfig schema: {exc}\n"
                f"Data keys: {list(data.keys())}"
            )

    def _validate_consistency(self, config: DomainConfig) -> None:
        """Raise ValueError if Groq introduced logical inconsistencies."""
        feature_names  = config.all_feature_names()
        proxy_names    = [f.name for f in config.proxy_features]
        hidden_names   = [f.name for f in config.hidden_features]
        explicit_names = [f.name for f in config.explicit_features]

        if config.approval_threshold_feature not in feature_names:
            raise ValueError(
                f"approval_threshold_feature '{config.approval_threshold_feature}' "
                f"is not in any feature list. Known: {feature_names}"
            )

        if config.discriminated_group_feature not in proxy_names:
            raise ValueError(
                f"discriminated_group_feature '{config.discriminated_group_feature}' "
                f"must be one of proxy_features names: {proxy_names}"
            )

        # Validate discriminated_group_value is in the proxy feature's categories
        dgf = next(
            (f for f in config.proxy_features
             if f.name == config.discriminated_group_feature),
            None,
        )
        if dgf and dgf.categories:
            if config.discriminated_group_value not in dgf.categories:
                # Auto-fix: use the first category value
                config.__dict__["discriminated_group_value"] = dgf.categories[0]

        chain = config.causal_chain
        if len(chain) < 2:
            raise ValueError("causal_chain must have at least 2 entries")

        if chain[0] not in proxy_names:
            raise ValueError(
                f"causal_chain[0] '{chain[0]}' must be a proxy feature name. "
                f"Proxy names: {proxy_names}"
            )

        if chain[1] not in hidden_names:
            raise ValueError(
                f"causal_chain[1] '{chain[1]}' must be a hidden feature name. "
                f"Hidden names: {hidden_names}"
            )

        if len(config.explicit_features) != 6:
            raise ValueError(
                f"explicit_features must have exactly 6 entries, got "
                f"{len(config.explicit_features)}"
            )

        if len(config.proxy_features) != 3:
            raise ValueError(
                f"proxy_features must have exactly 3 entries, got "
                f"{len(config.proxy_features)}"
            )

        if len(config.hidden_features) != 2:
            raise ValueError(
                f"hidden_features must have exactly 2 entries, got "
                f"{len(config.hidden_features)}"
            )

        # ── v2 field validation (only when populated by Groq) ─────────────────

        if config.seasonal_decoy_feature is not None:
            if config.seasonal_decoy_feature not in explicit_names:
                raise ValueError(
                    f"seasonal_decoy_feature '{config.seasonal_decoy_feature}' "
                    f"must be one of explicit_features names: {explicit_names}"
                )

        if config.seasonal_decoy_secondary_feature is not None:
            if config.seasonal_decoy_secondary_feature not in explicit_names:
                raise ValueError(
                    f"seasonal_decoy_secondary_feature "
                    f"'{config.seasonal_decoy_secondary_feature}' "
                    f"must be one of explicit_features names: {explicit_names}"
                )

        if config.legitimate_risk_feature is not None:
            if config.legitimate_risk_feature not in explicit_names:
                raise ValueError(
                    f"legitimate_risk_feature '{config.legitimate_risk_feature}' "
                    f"must be one of explicit_features names: {explicit_names}"
                )

        if config.legitimate_risk_threshold is not None and config.legitimate_risk_feature is not None:
            feat_spec = next(
                (f for f in config.explicit_features if f.name == config.legitimate_risk_feature),
                None,
            )
            if feat_spec and feat_spec.value_range:
                lo, hi = feat_spec.value_range
                if not (lo <= config.legitimate_risk_threshold <= hi):
                    # Auto-fix to 30% of range rather than raising
                    fixed = round(lo + 0.30 * (hi - lo), 2)
                    object.__setattr__(config, "legitimate_risk_threshold", fixed)

        if config.newly_non_compliant_features is not None:
            bad = [f for f in config.newly_non_compliant_features if f not in proxy_names]
            if bad:
                raise ValueError(
                    f"newly_non_compliant_features contains names not in "
                    f"proxy_features: {bad}. Proxy names: {proxy_names}"
                )
            if config.discriminated_group_feature not in config.newly_non_compliant_features:
                # Auto-fix: ensure discriminated feature is always in the list
                object.__setattr__(
                    config,
                    "newly_non_compliant_features",
                    list(config.newly_non_compliant_features) + [config.discriminated_group_feature],
                )


# ── Convenience function ───────────────────────────────────────────────────────

def load_domain(description: str, seed: int = 42) -> DomainConfig:
    """One-liner helper: create a GroqGraphGenerator and call generate_cached."""
    return GroqGraphGenerator().generate_cached(description, seed=seed)
