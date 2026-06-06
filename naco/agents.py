from __future__ import annotations

from typing import Callable

Strategy = str
LLMBackend = Callable[[str], str]

STRATEGY_PROMPTS = {
    "E": "Respond with empathic validation and emotional reflection.",
    "X": "Respond with exploratory probing to deepen client insight.",
    "N": "Respond with neutral verbalization mirroring client affect.",
    "S": "Respond with brief supportive listening and gentle encouragement.",
}


class ClientAgent:
    def __init__(self, llm: LLMBackend | None = None):
        self.llm = llm

    def generate(self, persona: str, history: str, reference_client: str) -> str:
        prompt = (
            f"Persona:\n{persona}\n\n"
            f"Dialogue history:\n{history}\n\n"
            f"Reference client utterance:\n{reference_client}\n\n"
            "Rewrite the client response naturally while preserving emotional trajectory."
        )
        if self.llm is None:
            return reference_client
        return self.llm(prompt)


class CounselorAgent:
    def __init__(self, llm: LLMBackend | None = None):
        self.llm = llm

    def generate_candidate(
        self,
        history: str,
        client_query: str,
        strategy: Strategy,
        rag_context: str = "",
        reference_counselor: str = "",
    ) -> str:
        strategy_hint = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["E"])
        rag_block = f"\nRetrieved knowledge:\n{rag_context}\n" if rag_context else ""
        prompt = (
            f"{strategy_hint}\n"
            f"Dialogue history:\n{history}\n"
            f"Client query:\n{client_query}\n"
            f"{rag_block}"
            f"Reference counselor response:\n{reference_counselor}\n"
            "Generate one professional counseling response."
        )
        if self.llm is None:
            prefix = {"E": "[Reflection]", "X": "[Probing]", "N": "[Neutral]", "S": "[Support]"}[strategy]
            base = reference_counselor or "I hear what you are going through."
            return f"{prefix} {base}"
        return self.llm(prompt)
