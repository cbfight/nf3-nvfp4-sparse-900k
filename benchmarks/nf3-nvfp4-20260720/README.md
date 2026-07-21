# Sanitized validation record — 2026-07-20

This directory contains publication-safe evidence for the two native NF3 +
sparse NVFP4-KV profiles. Private addresses, hostnames, usernames, local paths,
Hermes request content, and tool definitions are excluded.

`results.json` preserves the source pins, full serving knobs, measured rates,
deep-context results, memory minima, and opaque hashes for the private Hermes
capture. Rates use client-observed wall time.

Important definitions:

- `single_stream_median_tokens_per_second`: median of three 512-output-token
  non-streaming requests after one warmup.
- `concurrent_aggregate_tokens_per_second`: sum of completion tokens divided by
  concurrent batch wall time.
- `prefill_approximation_tokens_per_second`: prompt tokens divided by the full
  wall time of a unique long-prompt, one-output-token request.
- `streaming_decode_tokens_per_second`: tokens after the first visible token
  divided by wall time after that token.
- `ttft_seconds`: request start to first visible content or reasoning delta.

The prefill approximation includes scheduling and first-token work and must not
be represented as isolated kernel throughput.

Run the public harness with the concurrency declared by the profile:

```bash
python3 benchmarks/benchmark_openai.py \
  --base-url http://192.168.192.1:8210/v1 \
  --model glm-5.2 \
  --profile context900 \
  --concurrency 4 \
  --json-out context900-reproduction.json
```

Use `--concurrency 6 --profile fast262` for the fast lane. The published run
used a private real-skills corpus whose contents cannot be distributed. The
harness therefore defaults to a deterministic synthetic corpus; supply a
public file with `--prefill-file` when comparing across machines and report its
hash and resulting prompt-token count.
