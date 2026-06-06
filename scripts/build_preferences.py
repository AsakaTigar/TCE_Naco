#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from naco.data import export_preferences, load_json, sessions_to_preferences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Progressive-SimPO preference pairs.")
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/naco_corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("data/preferences/prog_simpo_pairs.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = load_json(args.input)
    pairs = sessions_to_preferences(sessions)
    count = export_preferences(pairs, args.output)
    print(f"Exported {count} preference pairs -> {args.output}")


if __name__ == "__main__":
    main()
