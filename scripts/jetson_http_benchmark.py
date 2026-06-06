import argparse
import json
import os
import queue
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests


def start_tegrastats(interval_ms: int, outfile: Path) -> subprocess.Popen:
    with outfile.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def iter_sse_lines(resp: requests.Response):
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith("data: "):
            data = raw[6:]
            if data.strip() == "[DONE]":
                break
            yield data


def count_output_tokens(text: str) -> int:
    return max(1, len(text.strip().split()))


def benchmark_once(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start = time.perf_counter()
    ttft = None
    chunks = []

    with requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for item in iter_sse_lines(resp):
            obj = json.loads(item)
            delta = obj["choices"][0]["delta"].get("content", "")
            if delta and ttft is None:
                ttft = time.perf_counter() - start
            if delta:
                chunks.append(delta)

    total_latency = time.perf_counter() - start
    output_text = "".join(chunks)
    output_tokens = count_output_tokens(output_text)
    decode_time = max(1e-6, total_latency - (ttft or total_latency))
    tok_s = output_tokens / decode_time

    return {
        "ttft_s": round(ttft or total_latency, 4),
        "total_latency_s": round(total_latency, 4),
        "output_tokens_est": output_tokens,
        "tokens_per_s_est": round(tok_s, 4),
        "output_preview": output_text[:160],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible local LLM endpoint on Jetson.")
    parser.add_argument("--base-url", required=True, help="Example: http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("jetson_benchmark"))
    args = parser.parse_args()

    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file
        else "You are a counseling assistant. Respond in 120 words with empathetic reflection and one exploratory question."
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tegra_log = args.out_dir / "tegrastats.log"
    result_json = args.out_dir / "results.json"

    tegra_proc = start_tegrastats(args.interval_ms, tegra_log)
    results = []
    try:
        for i in range(args.runs):
            print(f"run {i+1}/{args.runs}")
            results.append(
                benchmark_once(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            )
            time.sleep(2)
    finally:
        stop_process(tegra_proc)

    summary = {
        "runs": results,
        "mean_ttft_s": round(statistics.mean(x["ttft_s"] for x in results), 4),
        "mean_total_latency_s": round(statistics.mean(x["total_latency_s"] for x in results), 4),
        "mean_tokens_per_s_est": round(statistics.mean(x["tokens_per_s_est"] for x in results), 4),
        "prompt_chars": len(prompt),
        "tegrastats_log": str(tegra_log),
    }
    result_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
