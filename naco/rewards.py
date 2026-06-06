from __future__ import annotations

import re
from typing import Protocol

from naco.types import RewardBreakdown


class RewardScorer(Protocol):
    def score(self, history: str, client_query: str, candidate: str) -> RewardBreakdown: ...


class RuleBasedRewardScorer:
    """Fallback scorer used for offline reproduction without judge API."""

    EMPATHY_PATTERNS = (
        r"\b(understand|sounds like|must be|feel|hear you)\b",
        r"\b(validation|acknowledge|together)\b",
    )
    PROFESSIONAL_PATTERNS = (
        r"\b(CBT|DBT|reflection|probing|interpretation|reframe)\b",
        r"\b(technique|strategy|explore|pattern)\b",
    )
    EXPLORE_PATTERNS = (r"\?", r"\b(what|how|when|could|might)\b")
    UNSAFE_PATTERNS = (
        r"\b(kill yourself|suicide is|just ignore|stop taking)\b",
        r"\b(don't need therapy|you are fine)\b",
    )

    def score(self, history: str, client_query: str, candidate: str) -> RewardBreakdown:
        text = candidate.lower()
        empathy = self._pattern_score(text, self.EMPATHY_PATTERNS, base=2.5)
        professionalism = self._pattern_score(text, self.PROFESSIONAL_PATTERNS, base=2.3)
        exploration = self._pattern_score(text, self.EXPLORE_PATTERNS, base=2.0)
        safety = 1.0 if not any(re.search(p, text) for p in self.UNSAFE_PATTERNS) else 0.0
        safety *= 5.0
        return RewardBreakdown(
            empathy=min(empathy, 5.0),
            professionalism=min(professionalism, 5.0),
            exploration=min(exploration, 5.0),
            safety=safety,
        )

    @staticmethod
    def _pattern_score(text: str, patterns: tuple[str, ...], base: float) -> float:
        hits = sum(1 for p in patterns if re.search(p, text))
        return base + 0.6 * hits


class LLMRewardScorer:
    """LLM-as-judge interface; falls back to rules if backend unavailable."""

    def __init__(self, backend=None, fallback: RuleBasedRewardScorer | None = None):
        self.backend = backend
        self.fallback = fallback or RuleBasedRewardScorer()

    def score(self, history: str, client_query: str, candidate: str) -> RewardBreakdown:
        if self.backend is None:
            return self.fallback.score(history, client_query, candidate)
        try:
            return self.backend.score(history, client_query, candidate)
        except Exception:
            return self.fallback.score(history, client_query, candidate)
