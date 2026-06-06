from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCTSConfig:
    """CoRA-MCTS hyperparameters (Table in paper Sec. V)."""

    branching_factor: int = 4  # K
    simulations: int = 8  # T
    max_depth: int = 3  # d
    exploration_c: float = 1.0  # c
    expansion_threshold: int = 2  # N_thr
    temperature: float = 0.7
    strategies: tuple[str, ...] = ("E", "X", "N", "S")


@dataclass
class SynthesisConfig:
    persona_summary_max_tokens: int = 512
    max_turns: int = 8
    rag_top_k: int = 3
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


@dataclass
class SimPOConfig:
    beta: float = 0.1
    gamma: float = 0.5
    reference_free: bool = True


@dataclass
class TrainConfig:
    base_model: str = "/mnt/data/AODUOLI/models/Qwen2.5-7B-Instruct"
    output_dir: str = "results/naco_prog_simpo"
    num_epochs: int = 3
    per_device_batch_size: int = 2
    learning_rate: float = 5e-5
    max_length: int = 8192
    bf16: bool = True
    seed: int = 42
    simpo: SimPOConfig = field(default_factory=SimPOConfig)
