#!/usr/bin/env python3
"""Dependency-free client benchmark for an OpenAI-compatible GLM endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import pathlib
import statistics
import time
import urllib.error
import urllib.request

PROMPTS = [
    ("prose", "Continue writing a detailed technical essay about reproducible distributed systems. Use complete paragraphs and do not conclude until forced to stop."),
    ("code", "Implement a production-quality Python module for a bounded asynchronous work queue. Include type hints, cancellation, tests, and explanatory comments. Keep writing code until forced to stop."),
    ("structured", "Produce a numbered engineering incident timeline with concrete timestamps, observations, hypotheses, experiments, and conclusions. Continue adding entries until forced to stop."),
]


def body(model: str, prompt: str, tokens: int, stream: bool = False) -> dict:
    result = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"},
    }
    if stream:
        result.update({"stream": True, "stream_options": {"include_usage": True}})
    return result


def nonstream(url: str, model: str, label: str, prompt: str, tokens: int) -> dict:
    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(body(model, prompt, tokens)).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    elapsed = time.perf_counter() - started
    usage = result.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "label": label,
        "elapsed_seconds": round(elapsed, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_tokens_per_second": round(completion_tokens / elapsed, 3),
        "prompt_tokens_per_second": round(prompt_tokens / elapsed, 3),
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason", ""),
    }


def streaming(url: str, model: str, label: str, prompt: str, tokens: int) -> dict:
    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(body(model, prompt, tokens, True)).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    usage, finish = {}, ""
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
                continue
            chunk = json.loads(line[5:].strip())
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter()
                finish = choice.get("finish_reason") or finish
    ended = time.perf_counter()
    if first is None:
        raise RuntimeError(f"{label}: stream contained no visible token")
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_tokens, decode_seconds = max(0, completion_tokens - 1), ended - first
    return {
        "label": label,
        "elapsed_seconds": round(ended - started, 4),
        "time_to_first_token_seconds": round(first - started, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "end_to_end_tokens_per_second": round(completion_tokens / (ended - started), 3),
        "decode_tokens_per_second_excluding_first": round(decode_tokens / decode_seconds, 3) if decode_tokens else 0,
        "finish_reason": finish,
    }


def prefill_prompt(nonce: str, source_file: str | None, records: int) -> str:
    if source_file:
        corpus = pathlib.Path(source_file).read_text(encoding="utf-8")
    else:
        corpus = "\n".join(
            f"Reference record {i}: subsystem={i % 37}; value={i * 7919}; preserve during analysis."
            for i in range(records)
        )
    return f"Uncached prefill benchmark nonce: {nonce}\n\n{corpus}\n\nReply with one word: acknowledged."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prefill-file")
    parser.add_argument("--prefill-records", type=int, default=2200)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    url, nonce = args.base_url.rstrip("/"), f"{args.profile}-{time.time_ns()}"

    warmup = nonstream(url, args.model, "warmup", "Count upward with commas.", 32)
    singles = [nonstream(url, args.model, f"c1_{name}", text, args.decode_tokens) for name, text in PROMPTS]
    selected = [PROMPTS[i % len(PROMPTS)] for i in range(args.concurrency)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(nonstream, url, args.model, f"c{args.concurrency}_{i + 1}_{name}", text, args.decode_tokens) for i, (name, text) in enumerate(selected)]
        concurrent_samples = [future.result() for future in futures]
    concurrent_elapsed = time.perf_counter() - started

    prompt = prefill_prompt(nonce, args.prefill_file, args.prefill_records)
    prefill = nonstream(url, args.model, "unique_long_prefill", prompt, 1)
    streams = [streaming(url, args.model, f"stream_c1_{name}", text, args.decode_tokens) for name, text in PROMPTS]
    stream_prefill = streaming(url, args.model, "stream_unique_long_prefill", prefill_prompt(f"{nonce}-stream", args.prefill_file, args.prefill_records), 1)

    report = {
        "schema_version": 1,
        "benchmark_type": "decode_concurrency_prefill",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": args.profile,
        "notes": args.notes,
        "base_url": url,
        "model": args.model,
        "decode_tokens_requested": args.decode_tokens,
        "concurrency": args.concurrency,
        "prefill_source": args.prefill_file or f"generated:{args.prefill_records}-records",
        "warmup": warmup,
        "single_stream": singles,
        "single_stream_median_tokens_per_second": round(statistics.median(x["output_tokens_per_second"] for x in singles), 3),
        "concurrent": concurrent_samples,
        "concurrent_wall_seconds": round(concurrent_elapsed, 4),
        "concurrent_aggregate_tokens_per_second": round(sum(x["completion_tokens"] for x in concurrent_samples) / concurrent_elapsed, 3),
        "prefill": prefill,
        "streaming_single": streams,
        "streaming_single_median_decode_tokens_per_second_excluding_first": round(statistics.median(x["decode_tokens_per_second_excluding_first"] for x in streams), 3),
        "streaming_single_median_ttft_seconds": round(statistics.median(x["time_to_first_token_seconds"] for x in streams), 4),
        "streaming_prefill": stream_prefill,
        "measurement_notes": {
            "single": "completion tokens / full request wall time",
            "concurrent": "sum of completion tokens / batch wall time",
            "prefill": "prompt tokens / one-output-token wall time; not isolated kernel throughput",
            "streaming_decode": "tokens after first visible token / wall time after that token",
            "ttft": "request start to first visible content or reasoning delta"
        }
    }
    pathlib.Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "c1": report["single_stream_median_tokens_per_second"],
        "concurrent": report["concurrent_aggregate_tokens_per_second"],
        "prefill": prefill["prompt_tokens_per_second"],
        "stream_decode": report["streaming_single_median_decode_tokens_per_second_excluding_first"],
        "ttft": report["streaming_single_median_ttft_seconds"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
