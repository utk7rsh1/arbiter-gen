"""Validation test suite for the Groq-generated domain config pipeline.

Run with:
    pytest tests/test_groq_graph.py -v -s

All tests are tagged @pytest.mark.groq and require GROQ_API_KEY.
Without the key they are auto-skipped by conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbiter.env.groq_generator import GroqGraphGenerator
from arbiter.env.graph import generate_graph
from arbiter.env.environment import ArbiterEnv


# ── Test 1 — Parse and validate DomainConfig ──────────────────────────────────

@pytest.mark.groq
def test_groq_output_validates():
    """Groq output must pass schema validation and consistency checks."""
    gen    = GroqGraphGenerator()
    config = gen.generate_cached("A university admissions AI that screens applicants")

    assert len(config.explicit_features) == 6, (
        f"Expected 6 explicit features, got {len(config.explicit_features)}"
    )
    assert len(config.proxy_features) == 3, (
        f"Expected 3 proxy features, got {len(config.proxy_features)}"
    )
    assert len(config.hidden_features) == 2, (
        f"Expected 2 hidden features, got {len(config.hidden_features)}"
    )

    proxy_names = [f.name for f in config.proxy_features]
    assert config.discriminated_group_feature in proxy_names, (
        f"discriminated_group_feature '{config.discriminated_group_feature}' "
        f"not in proxy features: {proxy_names}"
    )

    assert config.causal_chain[0] in proxy_names, (
        f"causal_chain[0] '{config.causal_chain[0]}' must be a proxy feature"
    )

    hidden_names = [f.name for f in config.hidden_features]
    assert config.causal_chain[1] in hidden_names, (
        f"causal_chain[1] '{config.causal_chain[1]}' must be a hidden feature"
    )

    print(f"\n[test_groq_output_validates] Domain: {config.domain_name}")
    print(f"  Outcomes: {config.positive_outcome} / {config.negative_outcome}")
    print(f"  Threshold: {config.approval_threshold_feature} > {config.approval_threshold_value}")
    print(f"  Discriminated: {config.discriminated_group_value} in {config.discriminated_group_feature}")
    print(f"  Causal chain: {' → '.join(config.causal_chain)}")


# ── Test 2 — Graph generates without error ────────────────────────────────────

@pytest.mark.groq
def test_graph_generation():
    """generate_graph() must return correct keys and the right number of records."""
    gen    = GroqGraphGenerator()
    config = gen.generate_cached("A university admissions AI")
    ep_data = generate_graph(seed=42, domain=config)

    assert "graph"        in ep_data
    assert "records"      in ep_data
    assert "anomaly_info" in ep_data
    assert "features"     in ep_data
    assert len(ep_data["records"]) == 45, (
        f"Expected 45 records, got {len(ep_data['records'])}"
    )

    # All records must have an outcome matching the domain
    outcomes = {r["outcome"] for r in ep_data["records"]}
    assert outcomes.issubset({config.positive_outcome, config.negative_outcome}), (
        f"Unexpected outcome values: {outcomes - {config.positive_outcome, config.negative_outcome}}"
    )

    print(f"\n[test_graph_generation] Records: {len(ep_data['records'])}")
    print(f"  Anomaly type: {ep_data['anomaly_type']}")
    print(f"  Outcomes found: {outcomes}")


# ── Test 3 — Full episode runs without crash ──────────────────────────────────

@pytest.mark.groq
def test_full_episode():
    """A full episode (reset → 20 steps → done) must complete without exceptions."""
    gen    = GroqGraphGenerator()
    config = gen.generate_cached("A university admissions AI")
    env    = ArbiterEnv(level=1, seed=42, domain=config)

    obs = env.reset(seed=0)
    assert obs["budget_remaining"] == 20

    done = False
    for _ in range(20):
        obs, reward, done, info = env.step({"type": "QUERY_RECORDS", "feature_filter": {}})
        if done:
            break

    assert done, "Episode should be done after 20 steps (budget exhausted)"
    assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
    print(f"\n[test_full_episode] Final reward: {reward:.3f}")


# ── Test 4 — Multi-domain parametrised test ───────────────────────────────────

DOMAINS = [
    "A hiring AI that screens software engineering resumes",
    "A bank loan approval AI for small business loans",
    "A healthcare insurance claim approval system",
]


@pytest.mark.groq
@pytest.mark.parametrize("desc", DOMAINS)
def test_multi_domain(desc):
    """Each of three different domains must generate records successfully."""
    gen     = GroqGraphGenerator()
    config  = gen.generate_cached(desc)
    ep_data = generate_graph(seed=0, domain=config)

    assert len(ep_data["records"]) > 0, f"No records generated for: {desc}"

    features = ep_data["features"]
    assert len(features["explicit"]) == 6
    assert len(features["proxy"])    == 3
    assert len(features["hidden"])   == 2

    print(f"\n[test_multi_domain] {desc}")
    print(f"  Domain name: {config.domain_name}")
    print(f"  Records: {len(ep_data['records'])}  |  Anomaly: {ep_data['anomaly_type']}")


# ── Test 5 — Visual graph inspection (manual) ─────────────────────────────────

@pytest.mark.groq
def test_print_graph_summary():
    """Manual inspection test — run with -s and read the printed output."""
    gen    = GroqGraphGenerator()
    config = gen.generate_cached("A parole board decision AI")

    print(f"\n{'='*60}")
    print(f"Domain      : {config.domain_name}")
    print(f"Decision    : {config.positive_outcome} / {config.negative_outcome}")
    print(f"Description : {config.system_description[:120]}")
    print(f"\nExplicit features ({len(config.explicit_features)}):")
    for f in config.explicit_features:
        vr = f"  range={f.value_range}" if f.value_range else f"  cats={f.categories}"
        print(f"  {f.name:<30} [{f.dtype}]{vr}")
    print(f"\nProxy features ({len(config.proxy_features)}):")
    for f in config.proxy_features:
        print(f"  {f.name:<30} → protects: {f.protected_attribute}  cats={f.categories}")
    print(f"\nHidden features ({len(config.hidden_features)}):")
    for f in config.hidden_features:
        vr = f"  range={f.value_range}" if f.value_range else f"  cats={f.categories}"
        print(f"  {f.name:<30} [{f.dtype}]{vr}")
    print(f"\nThreshold   : {config.approval_threshold_feature} > {config.approval_threshold_value}")
    print(f"Discriminated: {config.discriminated_group_value!r} in {config.discriminated_group_feature}")
    print(f"Causal chain : {' → '.join(config.causal_chain)}")
    print(f"Anomaly desc : {config.anomaly_description[:120]}")
    print(f"{'='*60}")

    # Also run graph generation to confirm no crashes
    ep_data = generate_graph(seed=7, domain=config)
    assert len(ep_data["records"]) == 45


# ── Test 6 — Backwards compatibility: loan domain still works ─────────────────

def test_loan_domain_backwards_compat():
    """generate_graph(domain=None) must still work exactly as before."""
    ep_data = generate_graph(seed=99)

    assert "graph"        in ep_data
    assert "records"      in ep_data
    assert len(ep_data["records"]) == 45

    outcomes = {r["outcome"] for r in ep_data["records"]}
    assert outcomes.issubset({"approved", "denied"}), (
        f"Loan domain should only have approved/denied, got: {outcomes}"
    )

    print(f"\n[test_loan_domain_backwards_compat] Loan domain OK")
    print(f"  Anomaly type: {ep_data['anomaly_type']}")
    print(f"  Outcomes: {outcomes}")
