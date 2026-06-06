#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from naco.config import TrainConfig
from naco.data import load_json
from naco.progressive_simpo import ProgressiveSimPOLoss
from naco.types import PreferencePair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NaCo-Prog-SimPO.")
    parser.add_argument("--preferences", type=Path, default=Path("data/preferences/prog_simpo_pairs.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/naco_prog_simpo"))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_pairs(path: Path) -> list[PreferencePair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        PreferencePair(
            session_id=item["session_id"],
            turn_index=item["turn_index"],
            context=item["context"],
            chosen=item["chosen"],
            rejected=item["rejected"],
        )
        for item in raw
    ]


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if not args.device.startswith("cuda"):
        model = model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = ProgressiveSimPOLoss(cfg.simpo)
    pairs = load_pairs(args.preferences)
    if not pairs:
        raise SystemExit("No preference pairs found.")

    for step in range(args.max_steps):
        batch = pairs[step % len(pairs) : step % len(pairs) + 1]
        loss = loss_fn.batch_loss(model, batch, tokenizer, args.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
