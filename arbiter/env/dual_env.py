"""Level 7 — Dual-Auditor Co-Investigation (Halluminate sub-theme).

Two Auditors operate simultaneously on the same case:

Modes:
  - COLLABORATIVE: Claims broadcast between auditors; duplicate claims earn 0 reward;
    rewards pooled and split 50/50.
  - COMPETITIVE: First correct SUBMIT_REPORT takes 70% of terminal reward;
    second takes 30%. Speed matters.

Key mechanics:
  - Shared belief state: claim log, hypothesis flags, queried nodes.
  - Hypothesis divergence: if A marks hyp ELIMINATED and B marks it ACTIVE,
    Meta-Overseer deducts from both (-0.5 each).
  - One auditor (randomly chosen) is "biased" — pre-trained to over-favour Type 1.
    Extra proxy-discrimination records are injected into their data view so the
    bias is detectable through claim analysis by their partner.

Actions available at Level 7:
  BROADCAST_CLAIM   — explicitly re-broadcast a completed claim to the partner
                      (coordination tool; earns no additional reward).
  CHALLENGE_PARTNER — challenge the partner's Type-1 bias (+3.0 if correct,
                      -1.0 if wrong).
"""
from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from arbiter.env.environment import ArbiterEnv
from arbiter.env.schema_drift import _LOAN_REGULATION_NAME, _LOAN_CHANGED_FEATURES
from config import (
    DUAL_COMPETITIVE_FIRST_SHARE, DUAL_COMPETITIVE_SECOND_SHARE,
    REWARD_BIAS_DETECT_CORRECT, REWARD_CHALLENGE_WRONG,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_drift_alert(domain_context: Optional[Dict]) -> Dict:
    """Build the schema-change alert dict from domain context (or loan fallback)."""
    if domain_context:
        regulation          = domain_context.get("drift_regulation_name") or _LOAN_REGULATION_NAME
        newly_non_compliant = list(domain_context.get("newly_non_compliant_features") or [])
    else:
        regulation          = _LOAN_REGULATION_NAME
        newly_non_compliant = list(_LOAN_CHANGED_FEATURES)
    return {
        "regulation":          regulation,
        "newly_non_compliant": newly_non_compliant,
        "effective_immediately": True,
    }


# ── Shared state between two auditors in one dual episode ─────────────────────

class SharedEpisodeState:
    """Broadcast bus + conflict detector for a Level-7 dual-auditor episode."""

    def __init__(self, mode: str = "collaborative",
                 drift_step: Optional[int] = None):
        assert mode in ("collaborative", "competitive"), f"Unknown mode: {mode}"
        self.mode               = mode
        self.session_id         = str(uuid.uuid4())

        # Shared belief
        self.broadcast_claims:  List[Dict]  = []   # all claims from both auditors
        self.queried_nodes:     set         = set()
        self.hypothesis_flags:  Dict[str, Dict[str, str]] = {
            "A": {"proxy_discrimination": "ACTIVE",
                  "adversarial_injection": "ACTIVE",
                  "model_drift": "ACTIVE"},
            "B": {"proxy_discrimination": "ACTIVE",
                  "adversarial_injection": "ACTIVE",
                  "model_drift": "ACTIVE"},
        }

        # Schema drift coordination — fires once across the shared episode.
        self.drift_step:  Optional[int]  = drift_step
        self.drift_fired: bool           = False
        self.drift_alert: Optional[Dict] = None

        # Tracking
        self.divergence_violations: int = 0
        self.submitted: Dict[str, bool]  = {"A": False, "B": False}
        self.submit_order: List[str]     = []   # ["A", "B"] or ["B", "A"]
        self.rewards: Dict[str, float]   = {"A": 0.0, "B": 0.0}

        # Biased auditor assignment (random)
        self.biased_auditor: str = random.choice(["A", "B"])

    # ── Claim broadcasting ──────────────────────────────────────────────────────

    def register_claim(self, auditor_id: str, claim: Dict) -> Tuple[bool, float]:
        """
        Register a claim from an auditor.

        Duplicate detection is type-aware:
          causal          → match on cause_feature + effect_outcome
          counterfactual  → match on subject_record + counterfactual_feature
          theory_of_mind  → match on obfuscation_method + target_link

        Returns:
            (is_duplicate, duplicate_penalty)
        """
        claim_type = claim.get("claim_type", "causal")

        for bc in self.broadcast_claims:
            if bc["auditor"] == auditor_id:
                continue   # only cross-auditor duplicates matter
            if bc.get("claim_type") != claim_type:
                continue

            duplicate = False
            if claim_type == "causal":
                duplicate = (
                    bc.get("cause_feature")  == claim.get("cause_feature") and
                    bc.get("effect_outcome") == claim.get("effect_outcome")
                )
            elif claim_type == "counterfactual":
                duplicate = (
                    bc.get("subject_record")         == claim.get("subject_record") and
                    bc.get("counterfactual_feature") == claim.get("counterfactual_feature")
                )
            elif claim_type == "theory_of_mind":
                duplicate = (
                    bc.get("obfuscation_method") == claim.get("obfuscation_method") and
                    bc.get("target_link")        == claim.get("target_link")
                )

            if duplicate and self.mode == "collaborative":
                return True, 0.0

        self.broadcast_claims.append({"auditor": auditor_id, **claim})
        return False, 0.0

    # ── Hypothesis divergence ──────────────────────────────────────────────────

    def update_hypothesis(self, auditor_id: str,
                          hyp_type: str, status: str) -> float:
        """
        Update hypothesis flag for one auditor. Check for divergence.

        Returns divergence penalty (0.0 or -0.5) applied to BOTH auditors.
        """
        other = "B" if auditor_id == "A" else "A"
        self.hypothesis_flags[auditor_id][hyp_type] = status

        own_status   = status
        other_status = self.hypothesis_flags[other].get(hyp_type, "ACTIVE")

        diverged = (
            (own_status == "ELIMINATED" and other_status == "ACTIVE") or
            (own_status == "ACTIVE"     and other_status == "ELIMINATED")
        )
        if diverged:
            self.divergence_violations += 1
            penalty = -0.5
            self.rewards["A"] += penalty
            self.rewards["B"] += penalty
            return penalty
        return 0.0

    # ── Submit handling ─────────────────────────────────────────────────────────

    def register_submit(self, auditor_id: str, terminal_reward: float
                        ) -> Dict[str, float]:
        """
        Register a SUBMIT_REPORT and return the reward share for each auditor.

        Competitive: first = 70%, second = 30%
        Collaborative: 50% / 50%
        """
        self.submitted[auditor_id] = True
        self.submit_order.append(auditor_id)

        if self.mode == "collaborative":
            split = terminal_reward * 0.5
            self.rewards["A"] += split
            self.rewards["B"] += split
            return {"A": split, "B": split}
        else:  # competitive
            if len(self.submit_order) == 1:   # first to submit
                share = terminal_reward * DUAL_COMPETITIVE_FIRST_SHARE
            else:                              # second to submit
                share = terminal_reward * DUAL_COMPETITIVE_SECOND_SHARE
            self.rewards[auditor_id] += share
            return {auditor_id: share}

    # ── Bias detection reward ───────────────────────────────────────────────────

    def check_bias_challenge(self, challenger_id: str,
                             challenged_id: str,
                             claimed_bias_type: str) -> float:
        """
        Reward for correctly identifying the biased partner.
        +3.0 if correct, -1.0 if wrong.
        """
        if challenged_id == self.biased_auditor and claimed_bias_type == "type1_overfit":
            self.rewards[challenger_id] += REWARD_BIAS_DETECT_CORRECT
            return REWARD_BIAS_DETECT_CORRECT
        self.rewards[challenger_id] += REWARD_CHALLENGE_WRONG
        return REWARD_CHALLENGE_WRONG

    def to_observation_overlay(self, auditor_id: str,
                               is_biased: bool = False) -> Dict:
        """Return shared-state fields to inject into each auditor's observation."""
        other = "B" if auditor_id == "A" else "A"
        partner_claims = [c for c in self.broadcast_claims
                          if c["auditor"] == other]
        return {
            "dual_mode":              self.mode,
            "partner_claims":         partner_claims[-5:],  # last 5 from partner
            "partner_hypotheses":     self.hypothesis_flags[other],
            "divergence_violations":  self.divergence_violations,
            "biased_auditor_hint":    None,   # never revealed directly
            "is_biased_auditor":      is_biased,
            "shared_session_id":      self.session_id,
        }


# ── Dual-session environment ───────────────────────────────────────────────────

class DualArbiterEnv:
    """
    Wraps two ArbiterEnv instances sharing one episode graph via SharedEpisodeState.

    Usage:
        dual = DualArbiterEnv(level=7, mode="collaborative")
        obs_a, obs_b = dual.reset()
        obs_a, r_a, done, info = dual.step("A", action_a)
        obs_b, r_b, done, info = dual.step("B", action_b)
    """

    def __init__(self, level: int = 7, mode: str = "collaborative",
                 seed: Optional[int] = None,
                 domain: Optional[Any] = None):
        self.level  = level
        self.mode   = mode
        self.seed   = seed
        self.domain = domain   # DomainConfig | None — forwarded to both sub-envs

        # Both envs share the same underlying graph (same seed + domain).
        self.env_a = ArbiterEnv(level=level, seed=seed, domain=domain)
        self.env_b = ArbiterEnv(level=level, seed=seed, domain=domain)
        self.shared: Optional[SharedEpisodeState] = None
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset both environments and create a fresh shared state."""
        effective_seed = seed if seed is not None else self.seed
        obs_a = self.env_a.reset(seed=effective_seed)
        # env_b uses same seed so it generates the identical graph/anomaly.
        obs_b = self.env_b.reset(seed=effective_seed)

        # Synchronise the schema-drift step into shared state so the alert
        # fires exactly once across the shared episode, not once per auditor.
        drift_step = self.env_a._drift_step   # None when level < 6

        self.shared = SharedEpisodeState(mode=self.mode, drift_step=drift_step)
        self._done  = False

        # Inject extra proxy-discrimination records into the biased auditor's
        # data view so their Type-1 tendency is detectable via claim analysis.
        biased_env = self.env_a if self.shared.biased_auditor == "A" else self.env_b
        self._inject_type1_bias(biased_env)

        is_a_biased = (self.shared.biased_auditor == "A")
        obs_a["dual"] = self.shared.to_observation_overlay("A", is_biased=is_a_biased)
        obs_b["dual"] = self.shared.to_observation_overlay("B", is_biased=not is_a_biased)
        return obs_a, obs_b

    def step(self, auditor_id: str, action: Dict
             ) -> Tuple[Dict, float, bool, Dict]:
        """
        Step for one auditor.
        Returns (observation, reward, done, info) — same interface as ArbiterEnv.

        The returned `done` is True only when BOTH auditors have submitted.
        The returned `reward` is the auditor's actual share (split applied for
        SUBMIT_REPORT actions in competitive/collaborative mode).
        """
        assert auditor_id in ("A", "B"), "auditor_id must be 'A' or 'B'"
        assert self.shared is not None, "Call reset() first"

        env = self.env_a if auditor_id == "A" else self.env_b

        # Gracefully handle calls after this auditor has already submitted.
        if env._done:
            return (
                self._obs(auditor_id, env._observation()),
                0.0,
                self._all_done(),
                {"warning": "This auditor has already submitted. "
                            "Waiting for partner to finish."},
            )

        atype = action.get("type", "").upper()

        # ── Level-7 exclusive actions ─────────────────────────────────────────

        if atype == "BROADCAST_CLAIM":
            # Explicit re-broadcast of a completed claim to partner.
            # Earns no reward — purely a coordination move.
            claim      = action.get("claim", {})
            claim_type = action.get("claim_type", "causal")
            is_dup, _  = self.shared.register_claim(auditor_id, {
                **claim,
                "claim_type": claim_type,
                "broadcast":  True,
            })
            obs = self._obs(auditor_id, env._observation())
            return obs, 0.0, self._all_done(), {
                "broadcast_result": {"is_duplicate": is_dup, "claim_type": claim_type}
            }

        if atype == "CHALLENGE_PARTNER":
            challenged = "B" if auditor_id == "A" else "A"
            bias_type  = action.get("claimed_bias_type", "type1_overfit")
            reward = self.shared.check_bias_challenge(
                auditor_id, challenged, bias_type)
            obs = self._obs(auditor_id, env._observation())
            return obs, reward, self._all_done(), {"challenge_result": reward}

        # ── Hypothesis divergence check ───────────────────────────────────────

        if atype == "FLAG_HYPOTHESIS":
            div_pen = self.shared.update_hypothesis(
                auditor_id,
                action.get("hypothesis_type", ""),
                action.get("status", "ACTIVE"))
            obs_raw, reward, done, info = env.step(action)
            reward += div_pen
            if div_pen < 0:
                info["divergence_penalty"] = div_pen
        else:
            obs_raw, reward, done, info = env.step(action)

        # ── Claim broadcasting (collaborative duplicate suppression) ──────────

        if atype.startswith("CLAIM_"):
            claim      = action.get("claim", {})
            claim_type = atype.replace("CLAIM_", "").lower()
            is_dup, _  = self.shared.register_claim(auditor_id, {
                **claim,
                "claim_type": claim_type,
            })
            if is_dup and self.mode == "collaborative":
                reward = 0.0
                info["duplicate_claim"] = True

        # ── Terminal reward split ─────────────────────────────────────────────

        if atype == "SUBMIT_REPORT" and done:
            terminal_r = info.get("episode_reward", {}).get("total", 0.0)
            splits     = self.shared.register_submit(auditor_id, terminal_r)
            info["reward_split"] = splits
            # Return only this auditor's share, not the undivided total.
            reward = splits.get(auditor_id, reward)

        self.shared.queried_nodes.update(env._queried_nodes)

        # ── Shared schema-drift coordination ─────────────────────────────────
        # Fire the schema-change alert once for the whole shared episode
        # (whichever auditor first reaches the drift step triggers it).
        if (self.shared.drift_step is not None
                and not self.shared.drift_fired
                and env._step >= self.shared.drift_step):
            self.shared.drift_fired = True
            domain_ctx = getattr(env, "_domain_context", None)
            self.shared.drift_alert = _build_drift_alert(domain_ctx)

        # Strip the per-env alert (environment.py injects it per-step);
        # replace with the shared alert so both auditors see it together.
        obs_raw.pop("schema_change_alert", None)

        obs = self._obs(auditor_id, obs_raw)
        if self.shared.drift_fired and self.shared.drift_alert:
            obs["schema_change_alert"] = self.shared.drift_alert

        return obs, reward, self._all_done(), info

    def render(self, auditor_id: str = "A") -> Dict:
        env = self.env_a if auditor_id == "A" else self.env_b
        r = env.render()
        r["shared"] = {
            "broadcast_claims":      self.shared.broadcast_claims if self.shared else [],
            "divergence_violations": self.shared.divergence_violations if self.shared else 0,
            "mode":                  self.mode,
            "biased_auditor":        self.shared.biased_auditor if self.shared else None,
        }
        return r

    def get_metrics(self) -> Dict:
        return {
            "auditor_a": self.env_a.get_metrics(),
            "auditor_b": self.env_b.get_metrics(),
            "shared": {
                "mode":             self.mode,
                "broadcast_claims": len(self.shared.broadcast_claims) if self.shared else 0,
                "divergences":      self.shared.divergence_violations if self.shared else 0,
                "rewards":          self.shared.rewards if self.shared else {},
            },
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _obs(self, auditor_id: str, raw_obs: Dict) -> Dict:
        is_biased = (self.shared.biased_auditor == auditor_id) if self.shared else False
        raw_obs["dual"] = self.shared.to_observation_overlay(
            auditor_id, is_biased=is_biased)
        return raw_obs

    def _all_done(self) -> bool:
        return self.env_a._done and self.env_b._done

    def _inject_type1_bias(self, env: ArbiterEnv) -> None:
        """
        Augment the biased auditor's record set with 5 extra proxy-discrimination
        records so their Type-1 tendency is visible through claim analysis.
        """
        if env._ep is None:
            return

        records      = env._ep["records"]
        anomaly_info = env._ep["anomaly_info"]
        domain_ctx   = getattr(env, "_domain_context", None)

        # Resolve proxy feature id.
        proxy0_id = (
            anomaly_info.get("proxy_feature") or
            anomaly_info.get("post_drift_cause") or
            (env._ep["features"].get("proxy") or [""])[0]
        )

        # Resolve discriminated value from affected_demographic (format: "<feat>_<val>").
        affected  = anomaly_info.get("affected_demographic", "")
        disc_val  = (
            affected.split(f"{proxy0_id}_", 1)[-1]
            if proxy0_id and affected.startswith(f"{proxy0_id}_")
            else affected
        )

        negative_outcome = (
            domain_ctx.get("negative_outcome", "denied") if domain_ctx else "denied"
        )

        if not records or not proxy0_id or not disc_val:
            return

        for i in range(5):
            base = random.choice(records).copy()
            base["feature_vector"] = dict(base.get("feature_vector", {}))
            base["proxy_vector"]   = {
                **base.get("proxy_vector", {}), proxy0_id: disc_val
            }
            base["hidden_vector"]      = dict(base.get("hidden_vector", {}))
            base["id"]                 = f"rec_bias_{i:03d}"
            base["outcome"]            = negative_outcome
            base["is_anomalous"]       = True
            base["is_biased_injection"] = True
            records.append(base)

        env._ep["records"] = records


# ── Session registry for dual sessions ────────────────────────────────────────

_dual_sessions: Dict[str, DualArbiterEnv] = {}


def create_dual_session(level: int = 7, mode: str = "collaborative",
                        seed: Optional[int] = None,
                        domain: Optional[Any] = None) -> str:
    sid = str(uuid.uuid4())
    _dual_sessions[sid] = DualArbiterEnv(level=level, mode=mode, seed=seed, domain=domain)
    return sid


def get_dual_session(session_id: str) -> Optional[DualArbiterEnv]:
    return _dual_sessions.get(session_id)


def list_dual_sessions() -> List[str]:
    return list(_dual_sessions.keys())
