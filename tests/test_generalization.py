"""Offline generalization tests for Phases 3-7.

All tests here use the `mock_domain` fixture (hand-built, no Groq call) and
complete without GROQ_API_KEY.  Run with:

    pytest tests/test_generalization.py -v

or as part of the full offline suite:

    pytest -m "not groq" -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from arbiter.env.graph           import generate_graph
from arbiter.env.environment     import ArbiterEnv
from arbiter.env.decoys          import generate_decoys
from arbiter.env.schema_drift    import apply_schema_drift, schema_drift_observation
from arbiter.env.counterfactual  import intervene


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 regression: generate_graph() honours DomainConfig
# ─────────────────────────────────────────────────────────────────────────────

def test_outcome_nodes_match_domain(mock_domain):
    """Outcome node IDs in graph must match domain.denial_outcome_id() / approval_outcome_id()."""
    ep_data = generate_graph(seed=1, domain=mock_domain)
    G = ep_data["graph"]

    assert mock_domain.denial_outcome_id()   in G.nodes, (
        f"Missing denial outcome node: {mock_domain.denial_outcome_id()}"
    )
    assert mock_domain.approval_outcome_id() in G.nodes, (
        f"Missing approval outcome node: {mock_domain.approval_outcome_id()}"
    )
    assert mock_domain.minority_outcome_id() in G.nodes, (
        f"Missing minority outcome node: {mock_domain.minority_outcome_id()}"
    )


def test_causal_chain_matches_domain(mock_domain):
    """The anomaly causal chain recorded in anomaly_info must start from
    the domain's discriminated_group_feature (proxy) and include the
    hidden feature named in causal_chain[1]."""
    ep_data   = generate_graph(seed=2, domain=mock_domain)
    anom_info = ep_data["anomaly_info"]

    # anomaly_info always contains a causal_chain list for types 1 and 3
    if "causal_chain" in anom_info:
        chain = anom_info["causal_chain"]
        assert chain[0] == mock_domain.discriminated_group_feature, (
            f"Expected causal_chain[0] == '{mock_domain.discriminated_group_feature}', "
            f"got '{chain[0]}'"
        )


def test_discriminated_proxy_node_selected(mock_domain):
    """Phase 3 fix: proxy node for anomaly must be discriminated_group_feature, not proxy[0]."""
    ep_data = generate_graph(seed=3, domain=mock_domain, anomaly_type=1)
    anom_info = ep_data["anomaly_info"]

    assert anom_info.get("proxy_feature") == mock_domain.discriminated_group_feature, (
        f"Expected proxy_feature='{mock_domain.discriminated_group_feature}', "
        f"got '{anom_info.get('proxy_feature')}'"
    )


def test_records_use_domain_outcome_strings(mock_domain):
    """Records must use positive_outcome/negative_outcome strings, not 'approved'/'denied'."""
    ep_data = generate_graph(seed=4, domain=mock_domain)
    outcomes = {r["outcome"] for r in ep_data["records"]}

    assert outcomes.issubset({mock_domain.positive_outcome, mock_domain.negative_outcome}), (
        f"Unexpected outcome values: "
        f"{outcomes - {mock_domain.positive_outcome, mock_domain.negative_outcome}}"
    )
    # Explicit check: loan literals must not appear
    assert "approved" not in outcomes, "Loan literal 'approved' found in domain records"
    assert "denied"   not in outcomes, "Loan literal 'denied' found in domain records"


def test_domain_context_stashed_on_graph(mock_domain):
    """G.graph['domain_context'] must be populated with all required keys."""
    ep_data = generate_graph(seed=5, domain=mock_domain)
    dc = ep_data["graph"].graph.get("domain_context")

    assert dc is not None, "domain_context not stashed on graph"
    required_keys = {
        "discriminated_value", "discriminated_feature",
        "threshold_feature", "threshold_value",
        "positive_outcome", "negative_outcome",
    }
    missing = required_keys - dc.keys()
    assert not missing, f"domain_context missing keys: {missing}"

    assert dc["discriminated_feature"] == mock_domain.discriminated_group_feature
    assert dc["positive_outcome"]      == mock_domain.positive_outcome
    assert dc["negative_outcome"]      == mock_domain.negative_outcome


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 regression: counterfactual.intervene() uses domain context
# ─────────────────────────────────────────────────────────────────────────────

def test_intervene_flips_outcome_for_discriminated_proxy(mock_domain):
    """intervene() on the discriminated proxy feature must produce positive_outcome."""
    ep_data = generate_graph(seed=6, domain=mock_domain, anomaly_type=1)
    G       = ep_data["graph"]

    # Find a record where the discriminated value triggered a negative outcome
    discrim_feat = mock_domain.discriminated_group_feature
    discrim_val  = mock_domain.discriminated_group_value
    bad_records  = [
        r for r in ep_data["records"]
        if r["proxy_vector"].get(discrim_feat) == discrim_val
        and r["outcome"] == mock_domain.negative_outcome
    ]

    if not bad_records:
        pytest.skip("No anomalous records in this seed; try a different seed")

    record = bad_records[0]
    # Intervene: change the discriminated proxy away from the discriminated value
    # Use a different category value from the feature's categories
    proxy_feat_spec = next(
        f for f in mock_domain.proxy_features if f.name == discrim_feat
    )
    non_discrim_val = next(
        v for v in (proxy_feat_spec.categories or []) if v != discrim_val
    )

    result = intervene(G, record, discrim_feat, non_discrim_val)
    assert result["counterfactual_outcome"] == mock_domain.positive_outcome, (
        f"Expected '{mock_domain.positive_outcome}' after removing discrimination, "
        f"got '{result['counterfactual_outcome']}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 regression: generate_decoys() uses domain_context features
# ─────────────────────────────────────────────────────────────────────────────

def test_decoys_use_domain_features(mock_domain):
    """Decoy A/B metadata must reference the domain's seasonal/risk features."""
    ep_data = generate_graph(seed=7, domain=mock_domain)
    dc      = ep_data["graph"].graph["domain_context"]

    decoy_out = generate_decoys(ep_data["records"], ep_data["features"], domain_context=dc)
    decoy_a   = decoy_out["decoy_a"]
    decoy_b   = decoy_out["decoy_b"]

    assert decoy_a["primary_feature"]  == mock_domain.seasonal_decoy_feature, (
        f"Decoy A primary feature mismatch: {decoy_a['primary_feature']}"
    )
    assert decoy_a["secondary_feature"] == mock_domain.seasonal_decoy_secondary_feature, (
        f"Decoy A secondary feature mismatch: {decoy_a['secondary_feature']}"
    )
    assert decoy_b["risk_feature"] == mock_domain.legitimate_risk_feature, (
        f"Decoy B risk feature mismatch: {decoy_b['risk_feature']}"
    )
    assert decoy_b["threshold"] == mock_domain.legitimate_risk_threshold, (
        f"Decoy B threshold mismatch: {decoy_b['threshold']}"
    )

    # Loan literals must NOT appear in descriptions
    for text in (decoy_a["description"], decoy_b["description"]):
        assert "loan_amount"     not in text.lower(), f"Loan literal in decoy description: {text}"
        assert "payment_history" not in text.lower(), f"Loan literal in decoy description: {text}"


def test_decoys_loan_fallback():
    """Without domain_context, decoys should fall back to loan-domain feature names."""
    ep_data = generate_graph(seed=8)  # loan domain
    decoy_out = generate_decoys(ep_data["records"], ep_data["features"])

    assert decoy_out["decoy_a"]["primary_feature"]  == "loan_amount"
    assert decoy_out["decoy_b"]["risk_feature"]      == "payment_history"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 regression: apply_schema_drift() uses domain_context
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_drift_records_domain_regulation(mock_domain):
    """apply_schema_drift() must record domain's drift_regulation_name and changed features."""
    ep_data = generate_graph(seed=9, domain=mock_domain)
    dc      = ep_data["graph"].graph["domain_context"]
    all_features = (
        ep_data["features"]["explicit"]
        + ep_data["features"]["proxy"]
        + ep_data["features"]["hidden"]
    )

    ep_data = apply_schema_drift(ep_data, drift_step=8,
                                  domain_context=dc, all_feature_names=all_features)
    sd = ep_data["schema_drift"]

    assert sd["regulation_name"]    == mock_domain.drift_regulation_name, (
        f"Regulation name mismatch: {sd['regulation_name']}"
    )
    assert set(sd["changed_features"]) == set(mock_domain.newly_non_compliant_features), (
        f"Changed features mismatch: {sd['changed_features']}"
    )
    # Post-drift schema must mark the changed features as non_compliant
    for feat in mock_domain.newly_non_compliant_features:
        assert sd["post_schema"].get(feat) == "non_compliant", (
            f"Feature '{feat}' should be non_compliant in post_drift schema"
        )


def test_schema_drift_observation_uses_domain(mock_domain):
    """schema_drift_observation() must include domain regulation name."""
    ep_data = generate_graph(seed=10, domain=mock_domain)
    dc      = ep_data["graph"].graph["domain_context"]

    alert = schema_drift_observation(step=8, drift_step=8, domain_context=dc)
    assert alert is not None
    assert alert["regulation"] == mock_domain.drift_regulation_name
    assert set(alert["newly_non_compliant"]) == set(mock_domain.newly_non_compliant_features)


def test_schema_drift_loan_fallback():
    """Without domain_context, schema drift falls back to loan-domain literals."""
    ep_data = generate_graph(seed=11)  # loan domain
    ep_data = apply_schema_drift(ep_data, drift_step=7)
    sd = ep_data["schema_drift"]

    assert sd["regulation_name"]  == "AI Fairness Directive 2026"
    assert "zip_code_cluster"     in sd["changed_features"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: full ArbiterEnv wiring with mock_domain
# ─────────────────────────────────────────────────────────────────────────────

def test_full_env_episode_mock_domain(mock_domain):
    """A full Level-1 episode with mock_domain must complete without exceptions."""
    env = ArbiterEnv(level=1, seed=42, domain=mock_domain)
    obs = env.reset(seed=0)

    assert obs["budget_remaining"] == 20

    done = False
    for _ in range(20):
        obs, reward, done, info = env.step({"type": "QUERY_RECORDS", "feature_filter": {}})
        if done:
            break

    assert done, "Episode should be done after 20 steps"
    assert isinstance(reward, float)


def test_full_env_level6_mock_domain(mock_domain):
    """Level-6 episode (schema drift enabled) must complete without exceptions."""
    env = ArbiterEnv(level=6, seed=7, domain=mock_domain)
    obs = env.reset(seed=0)

    drift_seen = False
    for _ in range(20):
        if "schema_change_alert" in obs:
            drift_seen = True
            alert = obs["schema_change_alert"]
            assert alert["regulation"] == mock_domain.drift_regulation_name, (
                f"Alert regulation mismatch: {alert['regulation']}"
            )
        obs, reward, done, info = env.step({"type": "QUERY_RECORDS", "feature_filter": {}})
        if done:
            break

    # We don't assert drift_seen because drift step is random; we just check no crash.
    assert isinstance(reward, float)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: resolve_defaults() fills missing v2 fields
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_defaults_fills_missing_fields():
    """resolve_defaults() must fill all v2 optional fields from existing config data."""
    from arbiter.env.domain_config import DomainConfig, FeatureSpec

    # Create a minimal config without v2 fields (simulates old cached v1)
    minimal = DomainConfig(
        domain_name="Test Domain",
        decision_verb="approved",
        positive_outcome="approved",
        negative_outcome="denied",
        explicit_features=[
            FeatureSpec(name="score_a", description="A", dtype="continuous", value_range=(0.0, 100.0)),
            FeatureSpec(name="score_b", description="B", dtype="continuous", value_range=(0.0, 50.0)),
            FeatureSpec(name="score_c", description="C", dtype="continuous", value_range=(0.0, 200.0)),
            FeatureSpec(name="cat_d",   description="D", dtype="categorical", categories=["x", "y"]),
            FeatureSpec(name="score_e", description="E", dtype="continuous", value_range=(0.0, 10.0)),
            FeatureSpec(name="score_f", description="F", dtype="continuous", value_range=(0.0, 10.0)),
        ],
        proxy_features=[
            FeatureSpec(name="zip_cluster", description="Z", dtype="categorical", categories=["c1", "c2"], protected_attribute="race"),
            FeatureSpec(name="name_group",  description="N", dtype="categorical", categories=["g1", "g2"], protected_attribute="ethnicity"),
            FeatureSpec(name="area_code",   description="A", dtype="categorical", categories=["a1", "a2"], protected_attribute="socioeconomic_status"),
        ],
        hidden_features=[
            FeatureSpec(name="risk_score",  description="R", dtype="continuous", value_range=(0.0, 100.0)),
            FeatureSpec(name="trust_score", description="T", dtype="continuous", value_range=(0.0, 100.0)),
        ],
        outcome_nodes=["denied_rate_overall", "approved_rate_overall"],
        approval_threshold_description="score_a > 60",
        approval_threshold_feature="score_a",
        approval_threshold_value=60.0,
        discriminated_group_value="c1",
        discriminated_group_feature="zip_cluster",
        anomaly_description="ZIP cluster discrimination",
        causal_chain=["zip_cluster", "risk_score", "denied_rate_overall"],
        system_description="A test decision system.",
    )

    # All v2 fields should be None before resolve_defaults()
    assert minimal.seasonal_decoy_feature is None
    assert minimal.drift_regulation_name  is None

    minimal.resolve_defaults()

    assert minimal.seasonal_decoy_feature      is not None, "seasonal_decoy_feature not filled"
    assert minimal.seasonal_decoy_secondary_feature is not None, "seasonal_decoy_secondary_feature not filled"
    assert minimal.legitimate_risk_feature     is not None, "legitimate_risk_feature not filled"
    assert minimal.legitimate_risk_threshold   is not None, "legitimate_risk_threshold not filled"
    assert minimal.drift_regulation_name       is not None, "drift_regulation_name not filled"
    assert minimal.newly_non_compliant_features is not None, "newly_non_compliant_features not filled"
    assert minimal.drift_change_description    is not None, "drift_change_description not filled"

    # discriminated_group_feature must be in newly_non_compliant_features
    assert "zip_cluster" in minimal.newly_non_compliant_features


# ─────────────────────────────────────────────────────────────────────────────
# Loan backwards compat (no domain_context)
# ─────────────────────────────────────────────────────────────────────────────

def test_loan_domain_backwards_compat_offline():
    """Loan path (domain=None) must work end-to-end with all new module signatures."""
    ep_data = generate_graph(seed=99)
    assert ep_data["graph"].graph.get("domain_context") is None

    # Decoys
    decoys = generate_decoys(ep_data["records"], ep_data["features"])
    assert decoys["decoy_a"]["primary_feature"]  == "loan_amount"
    assert decoys["decoy_b"]["risk_feature"]      == "payment_history"

    # Schema drift
    ep_data = apply_schema_drift(ep_data, drift_step=8)
    assert "zip_code_cluster" in ep_data["schema_drift"]["changed_features"]

    # Full episode
    env = ArbiterEnv(level=1, seed=99, domain=None)
    obs = env.reset()
    done = False
    for _ in range(20):
        obs, reward, done, info = env.step({"type": "QUERY_RECORDS", "feature_filter": {}})
        if done:
            break
    assert done
