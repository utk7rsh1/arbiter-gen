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
from collections import defaultdict
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

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset_episode(self, level: Optional[int] = None):
        """Reset per-episode state."""
        if level is not None:
            self.level  = level
        self.budget     = OBFUSCATION_BUDGET.get(self.level, 0)
        self.action_log = []
        self.episode_count += 1

    def obfuscate(self, episode_data: Dict, hypothesis_flags: Optional[Dict] = None) -> Dict:
        """
        Apply obfuscations to the episode data (records + graph).
        Returns the modified episode_data and updates action_log.
        """
        if self.budget <= 0 or self.level < 4:
            return episode_data

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
        if self.level >= 5 and self.episode_count % DEFENDER_ADAPT_EVERY == 0 and any(self.freq_table.values()):
            # Adaptive: prefer methods that fooled the Auditor most
            total = sum(self.freq_table.values()) or 1
            population = list(self.freq_table.keys())
            weights    = [self.freq_table[m] / total for m in population]
            n = min(self.budget, 3)
            return self.rng.choices(population, weights=weights, k=n)
        else:
            # Rule-based: apply all methods in priority order
            priority = ["proxy_laundering", "link_substitution", "record_injection", "timestamp_manipulation"]
            return priority[:self.budget]

    # ── Obfuscation implementations ────────────────────────────────────────────

    def _link_substitution(self, data: Dict) -> Dict:
        """Replace a true causal edge with a plausible false one."""
        ainfo = data.get("anomaly_info", {})
        proxy = ainfo.get("proxy_feature") or ainfo.get("post_drift_cause", "zip_code_cluster")
        hidden = ainfo.get("hidden_mediator", "internal_risk_score")

        false_target = "fraud_history"
        action = {
            "method":      "link_substitution",
            "original":    f"{proxy}→{hidden}",
            "replacement": f"{proxy}→{false_target}",
            "target_link": f"{proxy}→{hidden}",
            "description": f"Replaced true edge {proxy}→{hidden} with plausible {proxy}→{false_target}",
        }
        self.action_log.append(action)

        # Mark the substitution in graph metadata
        if "graph" in data:
            G = data["graph"]
            if G.has_edge(proxy, hidden):
                G[proxy][hidden]["obfuscated"]    = True
                G[proxy][hidden]["false_target"]  = false_target
            G.add_edge(proxy, false_target,
                       edge_type="causal", true_causal=False, obfuscation="link_substitution", weight=0.60)

        return data

    def _record_injection(self, data: Dict) -> Dict:
        """Inject 5–8 synthetic records to dilute the anomaly signal."""
        n_inject = self.rng.randint(5, 8)
        ainfo    = data.get("anomaly_info", {})
        proxy    = ainfo.get("proxy_feature", "zip_code_cluster")

        injected = []
        existing = data.get("records", [])
        base_ts  = 0.5

        for i in range(n_inject):
            rec = {
                "id":             f"rec_inj_{i:03d}",
                "feature_vector": {"credit_score": self.rng.randint(580, 720),
                                   "debt_ratio":   round(self.rng.uniform(0.15, 0.40), 2),
                                   "employment_status": "employed"},
                "proxy_vector":   {proxy: self.rng.choice(["cluster_3", "cluster_4", "cluster_5"])},
                "hidden_vector":  {},
                "outcome":        "denied",           # injected records push benign demographics toward denial
                "timestamp":      base_ts + i * 0.01,
                "is_injected":    True,
                "is_anomalous":   False,
            }
            injected.append(rec)

        data["records"] = existing + injected

        action = {
            "method":      "record_injection",
            "target_link": f"statistical signal of {proxy}",
            "n_injected":  n_inject,
            "description": f"Injected {n_inject} synthetic records to dilute {proxy} signal",
        }
        self.action_log.append(action)
        return data

    def _proxy_laundering(self, data: Dict) -> Dict:
        """Insert an intermediate node between proxy and outcome."""
        ainfo  = data.get("anomaly_info", {})
        proxy  = ainfo.get("proxy_feature", "zip_code_cluster")
        hidden = ainfo.get("hidden_mediator", "internal_risk_score")

        intermediate = "neighborhood_commercial_viability"
        action = {
            "method":      "proxy_laundering",
            "target_link": f"{proxy}→{hidden}",
            "intermediate": intermediate,
            "description": (f"Laundered {proxy}→{hidden} via intermediate "
                            f"{proxy}→{intermediate}→business_viability_score→{hidden}"),
        }
        self.action_log.append(action)

        if "graph" in data:
            G = data["graph"]
            G.add_node(intermediate, name="Neighborhood Commercial Viability",
                       node_type="input_feature", proxy=True, hidden=False, obfuscation_node=True)
            if G.has_edge(proxy, hidden):
                G[proxy][hidden]["obfuscated"] = True
            G.add_edge(proxy, intermediate,   edge_type="causal", true_causal=False, weight=0.70)
            G.add_edge(intermediate, hidden,  edge_type="causal", true_causal=False, weight=0.75)

        return data

    def _timestamp_manipulation(self, data: Dict) -> Dict:
        """Shift the apparent drift timestamp (Type 3 only)."""
        ainfo     = data.get("anomaly_info", {})
        true_drift = ainfo.get("drift_timestamp", 0.5)
        fake_drift = true_drift + self.rng.uniform(0.10, 0.20)
        fake_drift = min(fake_drift, 0.85)

        action = {
            "method":       "timestamp_manipulation",
            "target_link":  "drift_timestamp",
            "true_value":   true_drift,
            "fake_value":   round(fake_drift, 2),
            "description":  f"Shifted drift timestamp from {true_drift} to {round(fake_drift,2)}",
        }
        self.action_log.append(action)

        # Shift records near the true drift point
        for rec in data.get("records", []):
            if abs(rec["timestamp"] - true_drift) < 0.05:
                rec["timestamp"] += self.rng.uniform(0.05, 0.15)

        return data
