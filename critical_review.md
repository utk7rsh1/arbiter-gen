# ARBITER-Gen: Critical Review — Levels 4–7 Generalization

> Reviewed: April 25, 2026  
> Scope: Every `.py` file under `arbiter/env/`, `arbiter/training/`, `arbiter/server.py`, `config.py`, and `tests/`

---

## Executive Verdict

**The generalization work is substantially complete.** The domain-config adapter pattern (`DomainConfig → domain_context → fallback-to-loan`) is applied consistently across `graph.py`, `environment.py`, `decoys.py`, `defender.py`, `schema_drift.py`, `dual_env.py`, `counterfactual.py`, `sft_generator.py`, and `grpo_trainer.py`. The test suite in `test_level_generalization.py` (810 lines, 10 targeted fixes) covers the exact bugs one would expect from this kind of refactor.

**However, there are real issues remaining — some of which could bite during a live demo or training run.** They range from a hardcoded magic number that breaks non-loan domains at Level 7, to missing test coverage for domain-aware GRPO training, to config constants that should be centralized but aren't.

---

## Per-Level Breakdown

### Level 4 — Rule-Based Defender ✅ Mostly Generalized

| Area | Status | Detail |
|------|--------|--------|
| `defender.py` — `_link_substitution` | ✅ | Uses `domain_context` to pick false target from explicit features; falls back to `"fraud_history"` only when `ctx` is None |
| `defender.py` — `_record_injection` | ✅ | Builds feature vectors from existing records (domain-faithful); resolves `negative_outcome` from context |
| `defender.py` — `_proxy_laundering` | ✅ | Intermediate node named from `ctx["negative_outcome"]`; falls back to `"neighborhood_commercial_viability"` only for loan |
| `defender.py` — `_timestamp_manipulation` | ✅ | No domain-specific strings — purely numeric |
| Test coverage | ✅ | `TestDefenderDomainAwareness` in [test_level_generalization.py](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/tests/test_level_generalization.py#L94-L194) covers all 4 methods + loan fallback |

> [!TIP]
> **Minor:** `_record_injection` only populates the `proxy_vector` for the anomaly proxy (`proxy0_id`) but leaves the other 2 proxy features empty in injected records. A careful auditor agent could trivially detect injected records by checking for missing proxy fields. This is a domain-quality issue, not a generalization bug, but it weakens the Defender.

---

### Level 5 — Adaptive Defender ✅ Generalized

| Area | Status | Detail |
|------|--------|--------|
| `defender.py` — `_select_methods()` | ✅ | Frequency table (`freq_table`) is keyed by method name, not domain features |
| `environment.py` — `record_auditor_result()` call | ✅ | Called in [_handle_submit_report()](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/arbiter/env/environment.py#L357-L364) after episode end |
| `config.py` — `DEFENDER_ADAPT_EVERY` | ✅ | Centralized constant |
| Test coverage | ✅ | `TestLevel5FreqTable` — verifies freq table updates after wrong verdicts |

> [!NOTE]
> The `record_auditor_result` call only fires when `self.curriculum.defender_active and self.defender.action_log`. This means if the defender happened to take 0 actions (e.g., budget=0), no frequency update occurs. This is correct behavior, not a bug.

---

### Level 6 — Schema Drift ✅ Generalized

| Area | Status | Detail |
|------|--------|--------|
| `schema_drift.py` — `build_schema_versions()` | ✅ | Reads `newly_non_compliant_features` from `domain_context`; falls back to `_LOAN_PRE/POST_DRIFT` |
| `schema_drift.py` — `apply_schema_drift()` | ✅ | Resolves `regulation_name`, `changed_features`, `change_description` from context |
| `schema_drift.py` — `schema_drift_observation()` | ✅ | Returns domain-specific alert or loan fallback |
| `environment.py` — drift wiring | ✅ | Passes `domain_context` and `all_feature_names` through to all schema_drift functions |
| `domain_config.py` — v2 fields | ✅ | `drift_regulation_name`, `newly_non_compliant_features`, `drift_change_description` with `resolve_defaults()` |
| Test coverage | ✅ | `TestSchemaDriftAlert` — flat dict check, loan and custom domain, list type check |

> [!WARNING]
> **`_LOAN_PRE_DRIFT` / `_LOAN_POST_DRIFT` are incomplete for the loan domain.** They list 9 features but the loan domain has 11 features (6 explicit + 3 proxy + 2 hidden). The hidden features (`internal_risk_score`, `behavioral_score`) are missing from the schema dict. This means `get_current_schema()` returns an incomplete compliance map when `domain_context is None`. The Auditor would never see hidden features anyway, but a judge inspecting the `/explain` endpoint could notice.

---

### Level 7 — Dual-Auditor Co-Investigation ⚠️ Mostly Generalized, with Issues

| Area | Status | Detail |
|------|--------|--------|
| `dual_env.py` — `SharedEpisodeState` | ✅ | No domain-specific strings in claim broadcasting, hypothesis divergence, or reward splits |
| `dual_env.py` — `_build_drift_alert()` | ✅ | Reads from `domain_context`; falls back to loan literals |
| `dual_env.py` — `_inject_type1_bias()` | ⚠️ | See Issue #1 below |
| `dual_env.py` — `BROADCAST_CLAIM` handler | ✅ | Domain-agnostic |
| `dual_env.py` — `CHALLENGE_PARTNER` | ✅ | Domain-agnostic — checks `biased_auditor` ID, not domain features |
| `dual_env.py` — Reward splits | ✅ | Hardcoded shares (70/30 competitive, 50/50 collaborative) |
| `dual_env.py` — schema drift sync | ✅ | Shared `drift_fired` flag prevents double-alert |
| Test coverage | ✅ | 10 test classes covering all Level 7 mechanics |

---

## Critical Issues

### Issue #1 — `_inject_type1_bias()` doesn't pass `domain` to the sub-environments ⚠️

```python
# dual_env.py L239-240
self.env_a = ArbiterEnv(level=level, seed=seed)
self.env_b = ArbiterEnv(level=level, seed=seed)
```

`DualArbiterEnv.__init__()` does **not** accept or forward a `domain` parameter. Both sub-environments are always created with `domain=None`, meaning **Level 7 always runs on the loan domain**, even if the caller passes a domain config.

**Impact:** If you try to demo Level 7 on a hiring domain (or any Groq-generated domain), the dual-auditor episode will silently fall back to loan features. The `create_dual_session` function also lacks a `domain` parameter.

**Fix needed in:** [dual_env.py L232-240](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/arbiter/env/dual_env.py#L232-L240), [dual_env.py L467-470](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/arbiter/env/dual_env.py#L467-L470), [server.py L217-233](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/arbiter/server.py#L217-L233)

---

### Issue #2 — `config.py` constants not used in `dual_env.py` reward splits

```python
# config.py L42-43
DUAL_COMPETITIVE_FIRST_SHARE  = 0.70
DUAL_COMPETITIVE_SECOND_SHARE = 0.30
```

But in `dual_env.py`, the splits are hardcoded:

```python
# dual_env.py L181-183
share = terminal_reward * 0.70   # should be DUAL_COMPETITIVE_FIRST_SHARE
share = terminal_reward * 0.30   # should be DUAL_COMPETITIVE_SECOND_SHARE
```

And the bias challenge rewards:
```python
# dual_env.py L197-199
self.rewards[challenger_id] += 3.0   # should be REWARD_BIAS_DETECT_CORRECT
self.rewards[challenger_id] -= 1.0   # should be REWARD_CHALLENGE_WRONG
```

**Impact:** Config tuning has no effect on Level 7 reward mechanics. Not a generalization bug per se, but violates the "single source of truth" principle established by `config.py`.

---

### Issue #3 — `server.py` doesn't pass `domain` to `create_session()`

```python
# server.py L118-122
@app.post("/sessions", response_model=SessionResponse)
def create_session_endpoint(req: CreateSessionRequest):
    sid = create_session(level=req.level, seed=req.seed)
```

`CreateSessionRequest` has no `domain` field. There's no way to create a custom-domain session via the REST API. The `create_session()` function in `environment.py` accepts `domain`, but the server endpoint doesn't expose it.

**Impact:** The React frontend cannot create sessions on arbitrary domains through the API. You'd need to either:
1. Add a `domain_description` field to `CreateSessionRequest` and call Groq server-side, or
2. Add a `domain_json` field to pass a pre-generated `DomainConfig`.

This is a **frontend integration blocker** for the generalization demo.

---

### Issue #4 — `Decoy B` secondary feature threshold hardcoded at `0.45`

```python
# decoys.py L96
if fv.get(secondary_feat, 0) > 0.45:
    r["outcome"] = negative_outcome
```

This threshold is hardcoded to `0.45`, which makes sense for `debt_ratio` (0–1 range in loan domain) but is meaningless for, say, `interview_score` (0–10 range) or `technical_score` (0–100 range). In a hiring domain where `interview_score` values range from 0 to 10, virtually every record in the seasonal window gets forced to `rejected` because nearly all values exceed 0.45.

**Impact:** Decoy A produces ~100% denial rate in the seasonal window for most non-loan domains, making it trivially detectable instead of subtle. This undermines the decoy's purpose.

**Fix:** Read the secondary feature's `value_range` from `domain_context` and compute a proportional threshold (e.g., 45th percentile of the range).

---

### Issue #5 — `counterfactual.py` has a residual loan-domain substring match

```python
# counterfactual.py L199-203
not discriminated_feature and (
    "zip_code" in feature_id
    or "surname" in feature_id
    or "neighborhood" in feature_id
)
```

This block is only reached when `discriminated_feature` is empty (which happens when `domain_context` is None, i.e., the loan path). For custom domains, `discriminated_feature` is always populated from `domain_context`, so this branch is dead code. **Not a bug**, but worth noting — if `domain_context` ever has an empty `discriminated_feature`, the counterfactual engine would incorrectly apply loan-domain logic.

---

### Issue #6 — No domain-awareness in `grpo_trainer.py` system prompt

```python
# grpo_trainer.py L41-43
SYSTEM_PROMPT = """You are an expert AI auditor. Output exactly one JSON action per turn.
Available actions: QUERY_RECORDS, QUERY_FEATURE_DISTRIBUTION, ..."""
```

When training with `--domain`, the environment uses the custom domain, but the model receives a generic system prompt that doesn't mention the domain's features, outcomes, or context. The `sft_generator.py` correctly builds a domain-specific prompt via `build_system_prompt(domain)`, but the GRPO trainer doesn't use it.

**Impact:** During GRPO training on a non-loan domain, the model gets observations with hiring features but a system prompt that says nothing about them. This likely degrades training quality.

**Fix:** Import `build_system_prompt` from `sft_generator.py` and use it when `--domain` is set:
```python
if _domain:
    from arbiter.training.sft_generator import build_system_prompt
    system_prompt = build_system_prompt(_domain)
```

---

### Issue #7 — `DomainConfig.resolve_defaults()` secondary feature collision

```python
# domain_config.py L218
if candidate and candidate.name != self.seasonal_decoy_feature:
```

If the domain has exactly 2 continuous explicit features, `resolve_defaults()` correctly picks the second one. But if it has only 1 continuous explicit feature, the fallback at line 221 sets `seasonal_decoy_secondary_feature` to the same value as `seasonal_decoy_feature`. This means Decoy A inflates the same feature twice, which makes it a stronger signal (doubly inflated) rather than a subtle two-feature correlation. Not a crash, but a quality issue.

---

## Test Coverage Assessment

| Level | Test File | Tests | Coverage Quality |
|-------|-----------|-------|-----------------|
| L4-L5 | [test_level_generalization.py](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/tests/test_level_generalization.py) | `TestDefenderDomainAwareness` (4 tests), `TestLevel5FreqTable` (2 tests) | ✅ Good — verifies no loan strings leak, loan fallback preserved |
| L6 | [test_level_generalization.py](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/tests/test_level_generalization.py) | `TestSchemaDriftAlert` (3 tests) | ✅ Good — flat dict, custom regulation name, list type |
| L7 | [test_level_generalization.py](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/tests/test_level_generalization.py) | 6 test classes (broadcast, bias, splits, drift sync, duplicates, graceful done) | ✅ Thorough |
| L4-L7 smoke | [test_level_generalization.py](file:///c:/Users/thoma/OneDrive/Desktop/Scaler%20meta/arbiter-gen/tests/test_level_generalization.py) | `TestFullEpisodeSmoke` (L4, L5, L6, L7) | ✅ Uses hiring domain |

### Missing Test Coverage

| Gap | Severity | Detail |
|-----|----------|--------|
| Dual-env with custom domain | 🔴 High | No test creates a `DualArbiterEnv` with a `domain` param (because Issue #1 — it doesn't support one) |
| Counterfactual with custom domain | 🟡 Medium | No test verifies `intervene()` correctly resolves outcomes using domain context |
| Decoy A threshold on non-loan domain | 🟡 Medium | No test checks that Decoy A's `0.45` threshold produces reasonable denial rates on non-loan feature ranges (Issue #4) |
| GRPO training with `--domain` | 🟡 Medium | No test verifies the GRPO system prompt is domain-aware (Issue #6) |
| Server API domain passthrough | 🔴 High | No test creates a session via REST API with a domain config (Issue #3) |

---

## Summary Scorecard

| Component | Generalized? | Issues |
|-----------|:------------:|--------|
| `domain_config.py` | ✅ | Minor: secondary feature collision |
| `groq_generator.py` | ✅ | Clean |
| `graph.py` | ✅ | Clean |
| `environment.py` | ✅ | Clean |
| `decoys.py` | ⚠️ | Issue #4: hardcoded 0.45 threshold |
| `defender.py` | ✅ | Minor: incomplete proxy vectors in injected records |
| `schema_drift.py` | ✅ | Minor: incomplete loan pre/post drift dicts |
| `counterfactual.py` | ✅ | Dead loan-domain code (harmless) |
| `claims.py` | ✅ | Already domain-agnostic |
| `reward.py` | ✅ | Already domain-agnostic |
| `meta_overseer.py` | ✅ | Already domain-agnostic |
| `curriculum.py` | ✅ | Already domain-agnostic |
| `dual_env.py` | ❌ | **Issue #1**: no domain param |
| `openenv_wrapper.py` | ✅ | Accepts domain, passes through |
| `server.py` | ❌ | **Issue #3**: no domain in REST API |
| `sft_generator.py` | ✅ | `build_system_prompt(domain)` works correctly |
| `grpo_trainer.py` | ⚠️ | **Issue #6**: generic system prompt |

---

## Priority Fix Order

1. **Issue #1** — `DualArbiterEnv` domain support (blocks Level 7 on any non-loan domain)
2. **Issue #3** — Server API domain passthrough (blocks frontend integration)
3. **Issue #4** — Decoy A seasonal threshold (produces unrealistic episodes on non-loan domains)
4. **Issue #6** — GRPO trainer domain-aware system prompt (training quality)
5. **Issue #2** — Centralize dual-env constants (code quality)
6. **Issue #7** — Secondary feature collision (edge case)
