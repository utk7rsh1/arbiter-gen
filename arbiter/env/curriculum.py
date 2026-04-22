"""Seven-Level Curriculum with Auto-Advancement for ARBITER."""
from collections import deque
from typing import Optional
from config import LEVEL_THRESHOLDS, ADVANCE_WINDOW


class Curriculum:
    """
    Tracks episode rewards per level and auto-advances when the threshold
    is met over the rolling window.
    """

    def __init__(self, start_level: int = 1):
        self.level   = start_level
        self.max_level = 7
        self.history: deque = deque(maxlen=ADVANCE_WINDOW)
        self.level_history = {l: [] for l in range(1, 8)}

    def record(self, episode_reward: float) -> Optional[int]:
        """
        Record episode reward and auto-advance if threshold met.

        Returns new level if advanced, else None.
        """
        self.history.append(episode_reward)
        self.level_history[self.level].append(episode_reward)
        return self.auto_advance()

    def auto_advance(self) -> Optional[int]:
        """Check if we should advance; return new level or None."""
        if self.level >= self.max_level:
            return None
        if len(self.history) < ADVANCE_WINDOW:
            return None

        mean_reward = sum(self.history) / len(self.history)
        threshold   = LEVEL_THRESHOLDS.get(self.level, 999)

        if mean_reward >= threshold:
            self.level  += 1
            self.history = deque(maxlen=ADVANCE_WINDOW)  # reset window
            return self.level
        return None

    def get_stats(self) -> dict:
        mean = sum(self.history) / len(self.history) if self.history else 0.0
        return {
            "level":        self.level,
            "window_mean":  round(mean, 2),
            "threshold":    LEVEL_THRESHOLDS.get(self.level, 999),
            "episodes_in_window": len(self.history),
            "gap_to_advance": round(max(0, LEVEL_THRESHOLDS.get(self.level, 999) - mean), 2),
        }

    @property
    def defender_active(self) -> bool:
        return self.level >= 4

    @property
    def tom_claims_enabled(self) -> bool:
        return self.level >= 4

    @property
    def schema_drift_enabled(self) -> bool:
        return self.level >= 6

    @property
    def multi_auditor_enabled(self) -> bool:
        return self.level >= 7
