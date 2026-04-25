"""Pytest configuration for ARBITER tests.

Test tiers
----------
- Default (no marker): fully offline, no API key required.
- @pytest.mark.groq  : requires GROQ_API_KEY; auto-skipped when absent.

Run only offline tests:
    pytest -m "not groq" -v

Run everything (needs GROQ_API_KEY):
    pytest -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest

# Make the repo root importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbiter.env.domain_config import DomainConfig, FeatureSpec


# ── Custom marks ──────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "groq: test requires GROQ_API_KEY environment variable",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked @pytest.mark.groq when GROQ_API_KEY is absent."""
    if os.environ.get("GROQ_API_KEY"):
        return   # key present: run all tests

    skip_groq = pytest.mark.skip(reason="GROQ_API_KEY not set; skipping Groq-gated tests")
    for item in items:
        if item.get_closest_marker("groq"):
            item.add_marker(skip_groq)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_domain() -> DomainConfig:
    """A hand-built hiring-AI DomainConfig - no Groq call needed."""
    return DomainConfig(
        domain_name="Software Engineering Hiring AI",
        decision_verb="hired",
        positive_outcome="hired",
        negative_outcome="rejected",
        explicit_features=[
            FeatureSpec(name="technical_score",   description="Technical interview score",     dtype="continuous", value_range=(0.0, 100.0)),
            FeatureSpec(name="years_experience",  description="Years of relevant experience",  dtype="continuous", value_range=(0.0, 20.0)),
            FeatureSpec(name="education_level",   description="Highest degree earned",         dtype="categorical", categories=["high_school", "bachelors", "masters", "phd"]),
            FeatureSpec(name="github_activity",   description="GitHub contribution score",     dtype="continuous", value_range=(0.0, 100.0)),
            FeatureSpec(name="coding_test_score", description="Online coding challenge score", dtype="continuous", value_range=(0.0, 100.0)),
            FeatureSpec(name="references_score",  description="Reference check composite",     dtype="continuous", value_range=(0.0, 10.0)),
        ],
        proxy_features=[
            FeatureSpec(name="university_prestige", description="University ranking tier",        dtype="categorical", categories=["tier_1", "tier_2", "tier_3", "tier_4"], protected_attribute="socioeconomic_status"),
            FeatureSpec(name="residential_zip",     description="Applicant residential ZIP area", dtype="categorical", categories=["affluent", "middle", "low_income", "rural"],  protected_attribute="race"),
            FeatureSpec(name="name_origin",         description="Inferred name ethnicity signal", dtype="categorical", categories=["western", "hispanic", "asian", "african"],    protected_attribute="ethnicity"),
        ],
        hidden_features=[
            FeatureSpec(name="culture_fit_score",  description="Internal culture fit estimate", dtype="continuous", value_range=(0.0, 100.0)),
            FeatureSpec(name="risk_score",         description="Attrition risk estimate",        dtype="continuous", value_range=(0.0, 100.0)),
        ],
        outcome_nodes=[
            "rejected_rate_overall",
            "hired_rate_overall",
            "flag_rate_overall",
            "rejected_rate_minority",
            "hired_rate_majority",
        ],
        approval_threshold_description="technical_score > 70",
        approval_threshold_feature="technical_score",
        approval_threshold_value=70.0,
        discriminated_group_value="low_income",
        discriminated_group_feature="residential_zip",
        anomaly_description=(
            "Applicants from low-income ZIP codes are systematically rejected "
            "regardless of technical score, mediated by culture_fit_score."
        ),
        causal_chain=["residential_zip", "culture_fit_score", "rejected_rate_overall"],
        system_description=(
            "An automated pre-screening tool used by a large tech firm to filter "
            "software engineering candidates. It scores candidates on technical "
            "ability, past experience, and online coding challenges."
        ),
        # v2 fields for decoy/drift tests
        seasonal_decoy_feature="technical_score",
        seasonal_decoy_secondary_feature="github_activity",
        legitimate_risk_feature="coding_test_score",
        legitimate_risk_threshold=30.0,
        drift_regulation_name="EEOC Disparate Impact Rule 2026",
        newly_non_compliant_features=["residential_zip", "name_origin"],
        drift_change_description=(
            "Under the EEOC Disparate Impact Rule 2026, use of residential ZIP "
            "codes and name-origin signals in hiring decisions is prohibited."
        ),
    )


@pytest.fixture(scope="session")
def loan_domain():
    """Fixture that returns None — exercises the backwards-compatible loan path."""
    return None
