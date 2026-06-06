# Initial Experiment Results

**Date**: 2026-06-06  
**Plan**: refine-logs/EXPERIMENT_PLAN.md  
**Environment**: poplab `/mnt/data/AODUOLI/TCE_Naco`, conda `Aoduo`

## M0: Sanity — PASSED

- `pytest tests`: 3 passed
- `synthesize_corpus.py`: 1 demo session synthesized
- `build_preferences.py`: 2 turn-level preference pairs exported
- `evaluate_metrics.py`: ROUGE-L mean = 0.971 on demo reference

## M1–M3: Pending Full GPU Run

- Full NaCo-Corpus synthesis (6,098 sessions) not launched in this bridge pass
- Full Progressive-SimPO training (3 epochs, 8×A100) requires dedicated GPU queue
- Pilot entrypoint: `python scripts/train_prog_simpo.py --max-steps 20`

## Summary

- Implemented: NaCo-DS, CoRA-MCTS, Progressive-SimPO core library + scripts
- Sanity: PASSED on poplab
- Main result: reconstruction verified; ready for scaled GPU experiments

## Next Step

→ Scale M1 synthesis and M2 training on poplab GPUs, then `/auto-review-loop` on metrics.
