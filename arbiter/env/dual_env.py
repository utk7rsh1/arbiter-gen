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
    The other must detect this through claim analysis.

New actions (only available at Level 7):
  BROADCAST_CLAIM       — explicitly re-broadcast a completed claim to partner
  CHALLENGE_PARTNER     — challenge a partner's claim (earns +1.0 if the challenged
                          claim turns out to be wrong, -0.5 if it was correct)
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from arbiter.env.environment import ArbiterEnv
from arbiter.env.meta_overseer import check_consistency


# ── Shared state between two auditors in one dual episode ─────────────────────

class SharedEpisodeState:
    """Broadcast bus + conflict detector for a Level-7 dual-auditor episode."""

    def __init__(self, mode: str = "collaborative"):
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

        # Tracking
        self.divergence_violations: int = 0
        self.submitted: Dict[str, bool]  = {"A": False, "B": False}
        self.submit_order: List[str]     = []   # ["A", "B"] or ["B", "A"]
        self.rewards: Dict[str, float]   = {"A": 0.0, "B": 0.0}

        # Biased auditor assignment (random)
        import random
        self.biased_auditor: str = random.choice(["A", "B"])

    # ── Claim broadcasting ──────────────────────────────────────────────────────

    def register_claim(self, auditor_id: str, claim: Dict) -> Tuple[bool, float]:
        """
        Register a claim from an auditor.

        Returns:
            (is_duplicate, duplicate_penalty)
        """
        # Check if this claim was already made by the OTHER auditor
        for bc in self.broadcast_claims:
            if (bc["auditor"] != auditor_id
                    and bc["claim_type"] == claim.get("claim_type")
                    and bc.get("cause_feature") == claim.get("cause_feature")
                    and bc.get("effect_outcome") == claim.get("effect_outcome")):
                # Duplicate in collaborative mode → 0 reward
                if self.mode == "collaborative":
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
        Register a SUBMIT_REPORT and return reward splits.

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
                share = terminal_reward * 0.70
                self.rewards[auditor_id] += share
                return {auditor_id: share}
            else:                              # second to submit
                share = terminal_reward * 0.30
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
            self.rewards[challenger_id] += 3.0
            return 3.0
        self.rewards[challenger_id] -= 1.0
        return -1.0

    def to_observation_overlay(self, auditor_id: str) -> Dict:
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
                 seed: Optional[int] = None):
        self.level = level
        self.mode  = mode
        self.seed  = seed

        # Both envs share the same underlying graph (same seed)
        self.env_a = ArbiterEnv(level=level, seed=seed)
        self.env_b = ArbiterEnv(level=level, seed=seed)
        self.shared: Optional[SharedEpisodeState] = None
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset both environments and create a fresh shared state."""
        effective_seed = seed or self.seed
        obs_a = self.env_a.reset(seed=effective_seed)
        # env_b uses same seed so it generates the same graph
        obs_b = self.env_b.reset(seed=effective_seed)

        self.shared = SharedEpisodeState(mode=self.mode)
        self._done  = False

        obs_a["dual"] = self.shared.to_observation_overlay("A")
        obs_b["dual"] = self.shared.to_observation_overlay("B")
        return obs_a, obs_b

    def step(self, auditor_id: str, action: Dict
             ) -> Tuple[Dict, float, bool, Dict]:
        """
        Step for one auditor.
        Returns (observation, reward, done, info) — same interface as ArbiterEnv.
        """
        assert auditor_id in ("A", "B"), "auditor_id must be 'A' or 'B'"
        assert self.shared is not None, "Call reset() first"

        env = self.env_a if auditor_id == "A" else self.env_b

        atype = action.get("type", "").upper()

        # Handle dual-specific actions first
        if atype == "CHALLENGE_PARTNER":
            challenged = "B" if auditor_id == "A" else "A"
            bias_type  = action.get("claimed_bias_type", "type1_overfit")
            reward = self.shared.check_bias_challenge(
                auditor_id, challenged, bias_type)
            obs = self._obs(auditor_id, env._observation())
            return obs, reward, self._all_done(), {"challenge_result": reward}

        # Hypothesis divergence check
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

        # Claim broadcasting (collaborative duplicate check)
        if atype.startswith("CLAIM_"):
            claim = action.get("claim", {})
            is_dup, _ = self.shared.register_claim(auditor_id, {
                **claim,
                "claim_type": atype.replace("CLAIM_", "").lower(),
            })
            if is_dup and self.mode == "collaborative":
                reward = 0.0
                info["duplicate_claim"] = True

        # Terminal reward split
        if atype == "SUBMIT_REPORT" and done:
            terminal_r = info.get("episode_reward", {}).get("total", 0.0)
            splits = self.shared.register_submit(auditor_id, terminal_r)
            info["reward_split"] = splits

        self.shared.queried_nodes.update(env._queried_nodes)
        obs = self._obs(auditor_id, obs_raw)
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
            "shared":    {
                "mode":               self.mode,
                "broadcast_claims":   len(self.shared.broadcast_claims) if self.shared else 0,
                "divergences":        self.shared.divergence_violations if self.shared else 0,
                "rewards":            self.shared.rewards if self.shared else {},
            }
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _obs(self, auditor_id: str, raw_obs: Dict) -> Dict:
        raw_obs["dual"] = self.shared.to_observation_overlay(auditor_id)
        return raw_obs

    def _all_done(self) -> bool:
        return self.env_a._done and self.env_b._done


# ── Session registry for dual sessions ────────────────────────────────────────

_dual_sessions: Dict[str, DualArbiterEnv] = {}


def create_dual_session(level: int = 7, mode: str = "collaborative",
                        seed: Optional[int] = None) -> str:
    sid = str(uuid.uuid4())
    _dual_sessions[sid] = DualArbiterEnv(level=level, mode=mode, seed=seed)
    return sid


def get_dual_session(session_id: str) -> Optional[DualArbiterEnv]:
    return _dual_sessions.get(session_id)


def list_dual_sessions() -> List[str]:
    return list(_dual_sessions.keys())
