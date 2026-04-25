"""Defender Obfuscation Engine for ARBITER.

Levels 1–4: Rule-based obfuscations.
Level 5+:   Frequency-adaptive (learns which obfuscations fool the current Auditor).

Obfuscation actions:
  link_substitution    – replace a true causal edge with a plausible false one
  record_injection     – add 5–8 synthetic records to dilute the statistical signal
  proxy_laundering     – insert an intermediate node between proxy and outcome
  timestamp_manipulation – shift the apparent drift timestamp (Type 3 only)
"""
import random
import copy
from typing import Dict, List, Optional

from config import OBFUSCATION_BUDGET, DEFENDER_ADAPT_EVERY


class Defender:
    """
    The AI Decision System under audit. Has a per-episode obfuscation budget
    and optionally learns from the Auditor's failure modes (Level 5+).
    """

    def __init__(self, level: int = 1, seed: Optional[int] = None):
        self.level   = level
        self.budget  = OBFUSCATION_BUDGET.get(level, 0)
        self.rng     = random.Random(seed)
        self.action_log: List[Dict] = []

        # Frequency table: obfuscation_method → success_count (Level 5+)
        self.freq_table: Dict[str, int] = {
            "link_substitution":      0,
            "record_injection":       0,
            "proxy_laundering":       0,
            "timestamp_manipulation": 0,
        }
        self.episode_count = 0
        self._domain_context: Optional[Dict] = None   # stashed per episode by obfuscate()

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset_episode(self, level: Optional[int] = None):
        """Reset per-episode state."""
        if level is not None:
            self.level  = level
        self.budget         = OBFUSCATION_BUDGET.get(self.level, 0)
        self.action_log     = []
        self.episode_count += 1
        self._domain_context = None   # cleared until obfuscate() is called this episode

    def obfuscate(self, episode_data: Dict, hypothesis_flags: Optional[Dict] = None) -> Dict:
        """
        Apply obfuscations to the episode data (records + graph).
        Returns the modified episode_data and updates action_log.
        """
        if self.budget <= 0 or self.level < 4:
            return episode_data

        # Stash domain context so every obfuscation method can be domain-aware.
        self._domain_context = None
        if "graph" in episode_data:
            self._domain_context = episode_data["graph"].graph.get("domain_context")

        data = copy.deepcopy(episode_data)
        methods = self._select_methods(hypothesis_flags)

        for method in methods:
            if self.budget <= 0:
                break
            if method == "link_substitution":
                data = self._link_substitution(data)
            elif method == "record_injection":
                data = self._record_injection(data)
            elif method == "proxy_laundering":
                data = self._proxy_laundering(data)
            elif method == "timestamp_manipulation":
                data = self._timestamp_manipulation(data)
            self.budget -= 1

        return data

    def record_auditor_result(self, method: str, fooled: bool):
        """Level 5+: update frequency table after episode ends."""
        if fooled:
            self.freq_table[method] = self.freq_table.get(method, 0) + 1

    # ── Method selection ───────────────────────────────────────────────────────

    def _select_methods(self, hypothesis_flags: Optional[Dict]) -> List[str]:
        """Choose obfuscation methods for this episode."""
        if (self.level >= 5
                and self.episode_count % DEFENDER_ADAPT_EVERY == 0
                and any(self.freq_table.values())):
            # Adaptive: prefer methods that fooled the Auditor most
            total      = sum(self.freq_table.values()) or 1
            population = list(self.freq_table.keys())
            weights    = [self.freq_table[m] / total for m in population]
            n = min(self.budget, 3)
            return self.rng.choices(population, weights=weights, k=n)
        else:
            # Rule-based; deprioritise methods that target already-eliminated hypotheses.
            priority = ["proxy_laundering", "link_substitution",
                        "record_injection", "timestamp_manipulation"]
            if hypothesis_flags:
                if hypothesis_flags.get("proxy_discrimination") == "ELIMINATED":
                    priority = [m for m in priority if m != "proxy_laundering"]
                if hypothesis_flags.get("model_drift") == "ELIMINATED":
                    priority = [m for m in priority if m != "timestamp_manipulation"]
            return priority[:self.budget]

    # ── Obfuscation implementations ────────────────────────────────────────────

    def _link_substitution(self, data: Dict) -> Dict:
        """Replace a true causal edge with a plausible false one."""
        ainfo  = data.get("anomaly_info", {})
        proxy  = ainfo.get("proxy_feature") or ainfo.get("post_drift_cause", "zip_code_cluster")
        hidden = ainfo.get("hidden_mediator", "internal_risk_score")

        # Domain-aware false target: pick any explicit feature outside the causal chain.
        ctx = self._domain_context
        if ctx:
            explicit_feats = data.get("features", {}).get("explicit", [])
            causal_chain   = ainfo.get("causal_chain", [])
            candidates     = [f for f in explicit_feats
                              if f not in causal_chain and f != proxy and f != hidden]
            false_target   = candidates[0] if candidates else (
                explicit_feats[0] if explicit_feats else f"{proxy}_auxiliary"
            )
        else:
            false_target = "fraud_history"   # loan-domain fallback

        action = {
            "method":      "link_substitution",
            "original":    f"{proxy}→{hidden}",
            "replacement": f"{proxy}→{false_target}",
            "target_link": f"{proxy}→{hidden}",
            "description": (f"Replaced true edge {proxy}→{hidden} with "
                            f"plausible {proxy}→{false_target}"),
        }
        self.action_log.append(action)

        if "graph" in data:
            G = data["graph"]
            if G.has_edge(proxy, hidden):
                G[proxy][hidden]["obfuscated"]   = True
                G[proxy][hidden]["false_target"] = false_target
            G.add_edge(proxy, false_target,
                       edge_type="causal", true_causal=False,
                       obfuscation="link_substitution", weight=0.60)

        return data

    def _record_injection(self, data: Dict) -> Dict:
        """Inject 5–8 synthetic records to dilute the anomaly signal."""
        n_inject = self.rng.randint(5, 8)
        ainfo    = data.get("anomaly_info", {})
        proxy    = ainfo.get("proxy_feature", "zip_code_cluster")

        ctx = self._domain_context
        negative_outcome = ctx.get("negative_outcome", "denied") if ctx else "denied"

        # Build a domain-faithful feature vector template from existing records.
        existing        = data.get("records", [])
        sample_fvec: Dict = {}
        benign_proxy_vals: List = []

        if existing:
            sample      = self.rng.choice(existing)
            sample_fvec = dict(sample.get("feature_vector", {}))
            sample_pvec = dict(sample.get("proxy_vector", {}))
            # Collect non-discriminated proxy values for the anomaly proxy feature.
            disc_val = str(ainfo.get("injection_value",
                                     ainfo.get("affected_demographic", "")))
            all_proxy_vals = [
                r.get("proxy_vector", {}).get(proxy)
                for r in existing
                if proxy in r.get("proxy_vector", {})
            ]
            benign_proxy_vals = [
                v for v in all_proxy_vals
                if v is not None and str(v) != disc_val
            ]
        else:
            sample_pvec = {}

        if not benign_proxy_vals:
            # Fall back to the sample's own proxy value or a neutral sentinel.
            fallback = existing[0].get("proxy_vector", {}).get(proxy, "neutral") if existing else "neutral"
            benign_proxy_vals = [fallback]

        # Pre-build per-proxy-feature value pools so every proxy column is populated.
        other_proxy_pools: dict = {}
        for r in existing:
            for k, v in r.get("proxy_vector", {}).items():
                if k != proxy:
                    other_proxy_pools.setdefault(k, []).append(v)

        injected = []
        for i in range(n_inject):
            # Perturb numeric fields slightly; leave categoricals unchanged.
            fvec = {}
            for k, v in sample_fvec.items():
                if isinstance(v, (int, float)):
                    fvec[k] = round(v * self.rng.uniform(0.85, 1.15), 4)
                else:
                    fvec[k] = v

            # Populate ALL proxy columns (not just the anomaly proxy) so injected
            # records are indistinguishable from real ones by missing-field detection.
            pvec = dict(sample_pvec)   # start from a real record's proxy vector
            pvec[proxy] = self.rng.choice(benign_proxy_vals)
            for k, pool in other_proxy_pools.items():
                pvec[k] = self.rng.choice(pool)

            rec = {
                "id":             f"rec_inj_{i:03d}",
                "feature_vector": fvec,
                "proxy_vector":   pvec,
                "hidden_vector":  {},
                "outcome":        negative_outcome,
                "timestamp":      0.5 + i * 0.01,
                "is_injected":    True,
                "is_anomalous":   False,
            }
            injected.append(rec)

        data["records"] = existing + injected

        action = {
            "method":      "record_injection",
            "target_link": f"statistical signal of {proxy}",
            "n_injected":  n_inject,
            "description": (f"Injected {n_inject} synthetic records to dilute "
                            f"{proxy} signal"),
        }
        self.action_log.append(action)
        return data

    def _proxy_laundering(self, data: Dict) -> Dict:
        """Insert an intermediate node between proxy and outcome."""
        ainfo  = data.get("anomaly_info", {})
        proxy  = ainfo.get("proxy_feature", "zip_code_cluster")
        hidden = ainfo.get("hidden_mediator", "internal_risk_score")

        # Domain-aware intermediate node name.
        ctx = self._domain_context
        if ctx:
            neg = ctx.get("negative_outcome", "decision")
            intermediate       = f"{neg}_risk_intermediary"
            intermediate_label = f"{neg.replace('_', ' ').title()} Risk Intermediary"
        else:
            intermediate       = "neighborhood_commercial_viability"
            intermediate_label = "Neighborhood Commercial Viability"

        action = {
            "method":       "proxy_laundering",
            "target_link":  f"{proxy}→{hidden}",
            "intermediate": intermediate,
            "description":  (f"Laundered {proxy}→{hidden} via intermediate "
                             f"{proxy}→{intermediate}→{hidden}"),
        }
        self.action_log.append(action)

        if "graph" in data:
            G = data["graph"]
            G.add_node(intermediate, name=intermediate_label,
                       node_type="input_feature", proxy=True, hidden=False,
                       obfuscation_node=True)
            if G.has_edge(proxy, hidden):
                G[proxy][hidden]["obfuscated"] = True
            G.add_edge(proxy, intermediate,  edge_type="causal", true_causal=False, weight=0.70)
            G.add_edge(intermediate, hidden, edge_type="causal", true_causal=False, weight=0.75)

        return data

    def _timestamp_manipulation(self, data: Dict) -> Dict:
        """Shift the apparent drift timestamp (Type 3 only)."""
        ainfo      = data.get("anomaly_info", {})
        true_drift = ainfo.get("drift_timestamp", 0.5)
        fake_drift = min(true_drift + self.rng.uniform(0.10, 0.20), 0.85)

        action = {
            "method":      "timestamp_manipulation",
            "target_link": "drift_timestamp",
            "true_value":  true_drift,
            "fake_value":  round(fake_drift, 2),
            "description": (f"Shifted drift timestamp from {true_drift} "
                            f"to {round(fake_drift, 2)}"),
        }
        self.action_log.append(action)

        for rec in data.get("records", []):
            if abs(rec["timestamp"] - true_drift) < 0.05:
                rec["timestamp"] += self.rng.uniform(0.05, 0.15)

        return data
