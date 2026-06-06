# Jetson Nano Benchmark Notes

Use the Nano as an edge-device feasibility check, not as a flagship-phone proxy.

## What to report

- Context length limit: the largest prompt length that still completes without OOM or forced truncation.
- Average latency per turn: mean total latency over 3 to 5 runs.
- TTFT: time to first token.
- Throughput: generated tokens per second.
- Memory usage: peak RAM from `tegrastats`.
- Power: average and peak `POM_5V_IN` or `VDD_IN` from `tegrastats`.

## Recommended setup

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
mkdir -p ~/naco_bench
```

Run your local model service first. If it exposes an OpenAI-compatible endpoint:

```bash
python3 scripts/jetson_http_benchmark.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model your-model-name \
  --runs 3 \
  --max-tokens 128 \
  --out-dir ~/naco_bench/run1
```

## How to get context-length limits

Test several prompt sizes, for example 512, 1024, 1536, 2048 tokens.

Practical method:

1. Prepare prompts of increasing length.
2. Keep `max_tokens` fixed, for example `128`.
3. Record the largest prompt that finishes stably in 3 consecutive runs.
4. If the runtime truncates silently, report the effective accepted context rather than the configured value.

## How to answer the reviewer

For Jetson Nano, position the result as:

- a low-power edge feasibility test;
- a conservative lower bound for on-device deployment;
- evidence that the quantized model remains usable on constrained embedded hardware.

Do not frame Nano results as equivalent to Snapdragon 8 Gen 3 or Orin-class hardware.
