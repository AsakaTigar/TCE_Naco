from naco.config import MCTSConfig
from naco.cora_mcts import CoRAMCTSPlanner
from naco.rewards import RuleBasedRewardScorer


def dummy_generator(history, client_query, strategy, rag_context=""):
    return f"[{strategy}] I hear you and we can explore this together."


def test_cora_mcts_returns_response():
    planner = CoRAMCTSPlanner(
        config=MCTSConfig(simulations=4, branching_factor=4),
        scorer=RuleBasedRewardScorer(),
        candidate_generator=dummy_generator,
    )
    response, strategy = planner.plan("Client: I feel stressed.\n", "I cannot sleep.")
    assert response
    assert strategy in {"E", "X", "N", "S", "ROOT"}
