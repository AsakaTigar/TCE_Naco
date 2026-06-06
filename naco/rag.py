from __future__ import annotations

import json
from pathlib import Path


class CounselingRAG:
    """Lightweight retrieval over a local counseling knowledge corpus."""

    def __init__(self, corpus_path: Path | None = None):
        self.entries: list[dict] = []
        if corpus_path and corpus_path.exists():
            self.entries = json.loads(corpus_path.read_text(encoding="utf-8"))

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.entries:
            return ""
        scored = []
        q_tokens = set(query.lower().split())
        for entry in self.entries:
            text = f"{entry.get('title', '')} {entry.get('content', '')}".lower()
            overlap = len(q_tokens & set(text.split()))
            scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = []
        for _, entry in scored[:top_k]:
            if entry.get("content"):
                chunks.append(f"- {entry['title']}: {entry['content']}")
        return "\n".join(chunks)
