# NaCo Experiment Plan (Paper Reconstruction)

**Paper:** NaCo — CoRA-MCTS + Progressive-SimPO (IEEE TCE)  
**Base model:** Qwen2.5-7B-Instruct  
**Environment:** poplab `/mnt/data/AODUOLI/TCE_Naco`, conda `Aoduo`

## Milestones

### M0 — Sanity (MUST-RUN)
- Unit tests: CoRA-MCTS, NaCo-DS, preference unfolding
- Demo synthesis on `data/raw/sample_sessions.json`
- Preference pair export
- Metric script smoke eval

**Success:** all tests pass; synthetic corpus + preference JSON written.

### M1 — Data Synthesis (MUST-RUN, scaled)
- Input: hotline + PsyDT reference sessions
- Run `scripts/synthesize_corpus.py` with LLM backend
- Target: NaCo-Corpus-scale pilot (100 sessions) before full 6,098

**Success:** multi-turn sessions with strategy metadata; avg counselor turn latency logged.

### M2 — Progressive-SimPO Training (MUST-RUN)
- Build 18k-style preference pairs via `scripts/build_preferences.py`
- Train with `scripts/train_prog_simpo.py`
- Hyperparams: `beta=0.1`, `lr=5e-5`, 3 epochs, batch size 2/GPU

**Success:** checkpoint saved; training loss decreases on pilot set.

### M3 — Evaluation (MUST-RUN)
- PsyDT / CPsyCoun automatic metrics via `scripts/evaluate_metrics.py`
- Win-rate judge script (GPT-4o / DeepSeek-V3 API optional)
- Edge benchmark via `scripts/jetson_http_benchmark.py`

**Success:** metrics JSON produced; main method beats Qwen2.5-7B-Instruct baseline on pilot set.

### M4 — Ablations (NICE-TO-HAVE)
- w/o persona persistence
- w/o RAG
- w/o length normalization / margin
- rule-based reward vs LLM reward

## Run Order

1. `python scripts/sanity_check.py`
2. `python scripts/synthesize_corpus.py --input data/raw/...`
3. `python scripts/build_preferences.py`
4. `python scripts/train_prog_simpo.py --max-steps 1000`
5. `python scripts/evaluate_metrics.py ...`

## Compute Budget

| Stage | Estimate |
|-------|----------|
| M0 sanity | CPU only |
| M1 pilot synthesis (100 sessions) | ~8 GPU-hours |
| M2 training (3 epochs, 8×A100) | ~24 GPU-hours |
| M3 evaluation | ~2 GPU-hours |
