from __future__ import annotations

import json
from pathlib import Path

from naco.agents import ClientAgent, CounselorAgent
from naco.config import SynthesisConfig
from naco.cora_mcts import CoRAMCTSPlanner
from naco.rag import CounselingRAG
from naco.rewards import LLMRewardScorer
from naco.types import Session, Turn


class NaCoDataSynthesizer:
    """NaCo-DS: persona-persistent multi-turn counseling synthesis."""

    def __init__(
        self,
        config: SynthesisConfig,
        client_agent: ClientAgent | None = None,
        counselor_agent: CounselorAgent | None = None,
        rag: CounselingRAG | None = None,
        scorer: LLMRewardScorer | None = None,
    ):
        self.config = config
        self.client_agent = client_agent or ClientAgent()
        self.counselor_agent = counselor_agent or CounselorAgent()
        self.rag = rag or CounselingRAG()
        self.scorer = scorer or LLMRewardScorer()

    def build_persona(self, reference_session: dict) -> str:
        turns = reference_session.get("turns", [])
        summary_lines = []
        for turn in turns[:3]:
            summary_lines.append(f"Client: {turn.get('client', '')}")
            summary_lines.append(f"Counselor: {turn.get('counselor', '')}")
        init_emotion = reference_session.get("initial_emotion", "anxious, guarded")
        traits = reference_session.get("traits", "perfectionistic, self-critical")
        return (
            f"Initial emotion: {init_emotion}\n"
            f"Personality traits: {traits}\n"
            "Session summary:\n" + "\n".join(summary_lines)
        )

    def synthesize_session(self, reference_session: dict) -> Session:
        persona = self.build_persona(reference_session)
        session = Session(
            session_id=reference_session.get("session_id", "unknown"),
            persona=persona,
            metadata={"source": reference_session.get("source", "reference")},
        )
        history = ""
        planner = CoRAMCTSPlanner(
            config=self.config.mcts,
            scorer=self.scorer,
            candidate_generator=self._make_candidate_fn(reference_session),
        )
        for idx, ref_turn in enumerate(reference_session.get("turns", [])[: self.config.max_turns]):
            client_text = self.client_agent.generate(
                persona=persona,
                history=history,
                reference_client=ref_turn.get("client", ""),
            )
            rag_context = self.rag.retrieve(client_text, top_k=self.config.rag_top_k)
            counselor_text, strategy = planner.plan(history, client_text, rag_context)
            session.turns.append(Turn(client=client_text, counselor=counselor_text))
            history += f"Client: {client_text}\nCounselor: {counselor_text}\n"
            session.metadata[f"turn_{idx}_strategy"] = strategy
        return session

    def _make_candidate_fn(self, reference_session: dict):
        ref_turns = reference_session.get("turns", [])

        def _candidate(history: str, client_query: str, strategy: str, rag_context: str = "") -> str:
            ref_counselor = ""
            if ref_turns:
                ref_counselor = ref_turns[min(len(ref_turns) - 1, history.count("Counselor:"))].get(
                    "counselor", ""
                )
            return self.counselor_agent.generate_candidate(
                history=history,
                client_query=client_query,
                strategy=strategy,
                rag_context=rag_context,
                reference_counselor=ref_counselor,
            )

        return _candidate

    def synthesize_corpus(self, input_path: Path, output_path: Path) -> int:
        records = json.loads(input_path.read_text(encoding="utf-8"))
        output = []
        for record in records:
            session = self.synthesize_session(record)
            output.append(
                {
                    "session_id": session.session_id,
                    "persona": session.persona,
                    "turns": [{"client": t.client, "counselor": t.counselor} for t in session.turns],
                    "metadata": session.metadata,
                }
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(output)
