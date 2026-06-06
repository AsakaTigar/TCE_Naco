from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from naco.config import MCTSConfig
from naco.rewards import RewardScorer


@dataclass
class MCTSNode:
    history: str
    client_query: str
    strategy: str
    depth: int
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    response: str = ""

    def uct(self, exploration_c: float) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        exploit = self.value / self.visits
        explore = exploration_c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return exploit + explore


class CoRAMCTSPlanner:
    """CoRA-MCTS counselor planning (Algorithm 1 in paper)."""

    def __init__(
        self,
        config: MCTSConfig,
        scorer: RewardScorer,
        candidate_generator,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.scorer = scorer
        self.candidate_generator = candidate_generator
        self.rng = rng or random.Random(42)

    def plan(self, history: str, client_query: str, rag_context: str = "") -> tuple[str, str]:
        root = MCTSNode(history=history, client_query=client_query, strategy="ROOT", depth=0)
        for _ in range(self.config.simulations):
            leaf = self._select(root)
            if leaf.depth < self.config.max_depth and leaf.visits >= self.config.expansion_threshold:
                leaf = self._expand(leaf, rag_context)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)
        best = self._best_child(root)
        return best.response, best.strategy

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children:
            current = max(current.children, key=lambda c: c.uct(self.config.exploration_c))
            if current.visits == 0:
                break
        return current

    def _expand(self, node: MCTSNode, rag_context: str) -> MCTSNode:
        strategies = list(self.config.strategies)
        self.rng.shuffle(strategies)
        for strategy in strategies[: self.config.branching_factor]:
            response = self.candidate_generator(
                history=node.history,
                client_query=node.client_query,
                strategy=strategy,
                rag_context=rag_context if strategy in {"E", "X"} else "",
            )
            child = MCTSNode(
                history=node.history,
                client_query=node.client_query,
                strategy=strategy,
                depth=node.depth + 1,
                parent=node,
                response=response,
            )
            node.children.append(child)
        return node.children[0] if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        if not node.response:
            node.response = self.candidate_generator(
                history=node.history,
                client_query=node.client_query,
                strategy=self.rng.choice(list(self.config.strategies)),
                rag_context="",
            )
        breakdown = self.scorer.score(node.history, node.client_query, node.response)
        return breakdown.total

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.value / max(c.visits, 1))
