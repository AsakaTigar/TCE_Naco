#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "-m", "pytest", "tests", "-q"])
    run([sys.executable, "scripts/synthesize_corpus.py"])
    run([sys.executable, "scripts/build_preferences.py"])
    run([
        sys.executable,
        "scripts/evaluate_metrics.py",
        "--predictions",
        "data/synthetic/naco_corpus.json",
        "--references",
        "data/raw/sample_sessions.json",
        "--output",
        "results/sanity_eval.json",
    ])
    print("Sanity pipeline passed.")


if __name__ == "__main__":
    main()
