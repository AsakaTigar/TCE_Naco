#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from naco.data import load_json


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_sessions(predictions: list[dict], references: list[dict]) -> dict:
    ref_map = {s["session_id"]: s for s in references}
    scores = []
    for pred in predictions:
        ref = ref_map.get(pred["session_id"])
        if not ref:
            continue
        for p_turn, r_turn in zip(pred.get("turns", []), ref.get("turns", [])):
            scores.append(rouge_l(p_turn.get("counselor", ""), r_turn.get("counselor", "")))
    avg = sum(scores) / max(len(scores), 1)
    return {"rouge_l_mean": avg, "num_turns": len(scores)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthesized counseling outputs.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/eval_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_sessions(load_json(args.predictions), load_json(args.references))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
