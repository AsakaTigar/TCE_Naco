from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    client: str
    counselor: str


@dataclass
class Session:
    session_id: str
    persona: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    context: str
    chosen: str
    rejected: str
    turn_index: int
    session_id: str


@dataclass
class RewardBreakdown:
    empathy: float
    professionalism: float
    exploration: float
    safety: float

    @property
    def total(self) -> float:
        weights = (0.3, 0.3, 0.2, 0.2)
        scores = (self.empathy, self.professionalism, self.exploration, self.safety)
        return sum(w * s for w, s in zip(weights, scores))
