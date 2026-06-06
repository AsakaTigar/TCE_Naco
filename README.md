# NaCo (TCE)

Official release materials for **NaCo: Realizing Human-Like Naturalistic Psychological Counseling with CoRA-MCTS Planning and Progressive-SimPO Alignment**, accepted to **IEEE Transactions on Consumer Electronics (TCE)**.

[![Paper PDF](https://img.shields.io/badge/Paper-TCE_Naco.pdf-blue)](paper/TCE_Naco.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

NaCo is a framework for naturalistic psychological counseling on consumer electronics platforms. It combines:

- **NaCo-DS** — persona-persistent data synthesis with emotional trajectory simulation
- **CoRA-MCTS** — Chain-of-Thought and Retrieval-Augmented Monte Carlo Tree Search for counselor planning
- **Progressive-SimPO** — turn-level preference optimization for multi-turn dialogue alignment

This repository provides the **accepted manuscript**, a **paper-faithful code reconstruction**, reproducibility scripts, and edge-deployment benchmark artifacts.

> **Note:** Full-scale reproduction (6,098-session corpus + 8×A100 training) requires GPU time and optional LLM judge APIs. A sanity pipeline and pilot training entrypoints are included for verification.

## Repository Layout

```
TCE_Naco/
├── naco/                   # Core library (NaCo-DS, CoRA-MCTS, Progressive-SimPO)
├── scripts/                # Synthesis, training, evaluation, sanity check
├── configs/                # Paper-aligned hyperparameters
├── data/                   # Sample sessions, RAG corpus, synthetic outputs
├── refine-logs/            # Experiment plan (experiment-bridge)
├── paper/                  # Accepted manuscript (LaTeX + PDF + figures)
├── reproducibility/        # Precomputed search-efficiency summaries
├── edge/                   # Jetson Orin NX benchmark notes and results
├── tests/                  # Unit tests
├── CITATION.bib
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
conda activate Aoduo
cd /mnt/data/AODUOLI/TCE_Naco
pip install -e .
python scripts/sanity_check.py
```

Full pipeline (experiment-bridge order):

```bash
python scripts/synthesize_corpus.py --input data/raw/sample_sessions.json
python scripts/build_preferences.py
python scripts/train_prog_simpo.py --max-steps 20
python scripts/evaluate_metrics.py --predictions data/synthetic/naco_corpus.json --references data/raw/sample_sessions.json
```

### 1. Compile the paper

```bash
cd paper
pdflatex bare_jrnl_new_sample4.tex
bibtex bare_jrnl_new_sample4
pdflatex bare_jrnl_new_sample4.tex
pdflatex bare_jrnl_new_sample4.tex
```

Or open `paper/TCE_Naco.pdf` directly for the accepted version.

### 2. Reproduce CoRA-MCTS search-efficiency simulation (R1-3)

```bash
pip install -r requirements.txt
python scripts/cora_mcts_efficiency_simulation.py --out-dir reproducibility/search_efficiency
python scripts/plot_search_efficiency_science.py --in-dir reproducibility/search_efficiency
```

Precomputed outputs are already in `reproducibility/search_efficiency/`.

### 3. Edge deployment benchmark (R2-7)

See `edge/JETSON_BENCHMARK.md` for the Jetson workflow. Example HTTP benchmark:

```bash
python scripts/jetson_http_benchmark.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model your-model-name \
  --runs 3 \
  --max-tokens 128 \
  --out-dir edge/run1
```

Measured Orin NX summaries: `edge/summary_table.csv`, `edge/reviewer_fill.md`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026naco,
  title   = {{NaCo}: Realizing Human-Like Naturalistic Psychological Counseling with {CoRA-MCTS} Planning and Progressive-{SimPO} Alignment},
  author  = {Li, Aoduo and Lin, Peikai and Li, Jiancheng and Xian, Jiahao and Lv, Haoran and Dong, Yihang and Chen, Xuhang},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2026},
  note    = {Accepted.}
}
```

## Authors

Aoduo Li, Peikai Lin, Jiancheng Li, Jiahao Xian, Haoran Lv, Yihang Dong, Xuhang Chen

## Acknowledgments

This work was supported by Guangdong Basic and Applied Basic Research Foundation (Grant No. 2024A1515140010).

## License

Released under the [MIT License](LICENSE).
