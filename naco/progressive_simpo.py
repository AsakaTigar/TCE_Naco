from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from naco.config import SimPOConfig
from naco.types import PreferencePair


@dataclass
class ProgressiveSimPOLoss:
    config: SimPOConfig

    def token_logprob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = shift_labels.ne(-100).float()
        lengths = mask.sum(dim=-1).clamp_min(1.0)
        return (token_logp * mask).sum(dim=-1) / lengths

    def turn_loss(
        self,
        chosen_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_logits: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> torch.Tensor:
        r_chosen = self.config.beta * self.token_logprob(chosen_logits, chosen_labels)
        r_rejected = self.config.beta * self.token_logprob(rejected_logits, rejected_labels)
        logits = r_chosen - r_rejected - self.config.gamma
        return -F.logsigmoid(logits).mean()

    def batch_loss(self, model, pairs: list[PreferencePair], tokenizer, device: str) -> torch.Tensor:
        losses = []
        for pair in pairs:
            chosen_inputs = tokenizer(pair.context + pair.chosen, return_tensors="pt").to(device)
            rejected_inputs = tokenizer(pair.context + pair.rejected, return_tensors="pt").to(device)
            chosen_out = model(**chosen_inputs, labels=chosen_inputs["input_ids"])
            rejected_out = model(**rejected_inputs, labels=rejected_inputs["input_ids"])
            losses.append(
                self.turn_loss(
                    chosen_out.logits,
                    chosen_inputs["input_ids"],
                    rejected_out.logits,
                    rejected_inputs["input_ids"],
                )
            )
        return torch.stack(losses).mean()


def build_preference_pairs_from_session(session: dict, reject_generator=None) -> list[PreferencePair]:
    turns = session.get("turns", [])
    pairs: list[PreferencePair] = []
    history = ""
    for idx, turn in enumerate(turns):
        if idx == 0:
            history = f"Client: {turn['client']}\n"
            continue
        context = history
        chosen = turn["counselor"]
        rejected = (
            reject_generator(context, turn["client"])
            if reject_generator
            else f"I understand. You should try to relax and stay positive about {turn['client'][:40]}."
        )
        pairs.append(
            PreferencePair(
                context=context,
                chosen=chosen,
                rejected=rejected,
                turn_index=idx,
                session_id=session.get("session_id", "unknown"),
            )
        )
        history += f"Client: {turn['client']}\nCounselor: {turn['counselor']}\n"
    return pairs
