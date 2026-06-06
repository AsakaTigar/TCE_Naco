from __future__ import annotations

import json
from pathlib import Path

from naco.progressive_simpo import build_preference_pairs_from_session
from naco.types import PreferencePair


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sessions_to_preferences(sessions: list[dict]) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for session in sessions:
        pairs.extend(build_preference_pairs_from_session(session))
    return pairs


def export_preferences(pairs: list[PreferencePair], output_path: Path) -> int:
    payload = [
        {
            "session_id": p.session_id,
            "turn_index": p.turn_index,
            "context": p.context,
            "chosen": p.chosen,
            "rejected": p.rejected,
        }
        for p in pairs
    ]
    save_json(output_path, payload)
    return len(payload)
