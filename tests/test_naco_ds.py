import json
from pathlib import Path

from naco.config import SynthesisConfig
from naco.naco_ds import NaCoDataSynthesizer
from naco.rag import CounselingRAG


def test_synthesize_single_session(tmp_path: Path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    sample = json.loads(Path("data/raw/sample_sessions.json").read_text(encoding="utf-8"))
    input_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    synthesizer = NaCoDataSynthesizer(
        config=SynthesisConfig(),
        rag=CounselingRAG(Path("data/rag_corpus.json")),
    )
    count = synthesizer.synthesize_corpus(input_path, output_path)
    assert count == 1
    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(out[0]["turns"]) == 3
