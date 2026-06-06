#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from naco.config import SynthesisConfig
from naco.naco_ds import NaCoDataSynthesizer
from naco.rag import CounselingRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NaCo-DS with CoRA-MCTS synthesis.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/sample_sessions.json"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/naco_corpus.json"))
    parser.add_argument("--rag-corpus", type=Path, default=Path("data/rag_corpus.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesizer = NaCoDataSynthesizer(
        config=SynthesisConfig(),
        rag=CounselingRAG(args.rag_corpus),
    )
    count = synthesizer.synthesize_corpus(args.input, args.output)
    print(f"Synthesized {count} sessions -> {args.output}")


if __name__ == "__main__":
    main()
