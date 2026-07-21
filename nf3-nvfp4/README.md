# MXFP8/NVFP4/NF3 Hybrid with Sparse NVFP4 KV Cache on 4× DGX Spark

This directory contains the public, pinned recipe that produced a working
900,000-token GLM-5.2 endpoint on four DGX Sparks while simultaneously using:

- MXFP8 non-expert projections (online quantized at load time);
- native mixed NVFP4/NF3 routed-expert weight execution;
- `B12X_MLA_SPARSE` attention with GLM-5.2's full/shared indexer pattern;
- `nvfp4_ds_mla` KV storage;
- TP4/DCP4 distributed serving;
- three-token MTP speculative decoding; and
- GLM tool/reasoning parsers compatible with a 91-tool Hermes request.

This was validated on 2026-07-20. It is an experimental integration, not an
upstream-supported vLLM configuration.

## What this proves

The launch is not inferred from environment variables alone. The running
engine emitted both of these selections:

```text
Using AttentionBackendEnum.B12X_MLA_SPARSE backend.
Using nvfp4_ds_mla data type to store kv cache.
```

`verify-runtime.sh` checks the live command, environment, health endpoint, and
these log lines. NVFP4 does not make attention sparse: sparse B12X MLA chooses
which KV records to attend to, while `nvfp4_ds_mla` independently compresses
those records. Messages about skipping the sparse indexer on `F` layers are
expected because the checkpoint's pattern intentionally mixes full (`F`) and
sparse/shared (`S`) layers.

## Pinned stack

| Component | Pin |
|---|---|
| Checkpoint | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` at `c3b696ed315be5f8371695d308d72b0742b5d2f9` |
| vLLM fork | `local-inference-lab/vllm` at `70c49aad686417aa2f15123731971d56edb4ded6` |
| b12x | `voipmonitor/b12x` at `1377d5f22c98de0c17d9b3f35a5b56d7587992fa` |
| build harness | `m9e/blackwell-llm-docker` at `d3247230dfbd34b9467791c404a8ad2942c1b8e6` |
| PyTorch / Triton | `2.12.0+cu132` / `3.7.0` |
| FlashInfer | `0.6.14` |
| CUTLASS DSL | `4.5.2` |
| validated image ID | `sha256:786af9b760056578eb57869e66d27f43ccf6fd048cf220fa0e877278468618a7` |

The image ID is evidence from our build, not a portable requirement: a clean
rebuild may have a different layer digest while retaining the pinned sources.

## Hardware and network assumptions

- Four DGX Spark/GB10 systems, one GPU per node.
- The same checkpoint and image on all four nodes.
- A working RoCEv2 rail with jumbo MTU end to end.
- Passwordless SSH from the machine running `launch.sh` to each node.
- At least 8 GiB `MemAvailable` per node before launch; the validated memory
  gate after startup and under load was 4 GiB.
- Roughly 350 GiB of model storage per node plus build and JIT-cache space.

Interface names and GID indices are installation-specific. Discover them with
`ip link`, `ibdev2netdev`, `ibv_devinfo`, and `show_gids`; do not copy the
example names blindly.

## 1. Download the exact checkpoint on every node

```bash
MODEL_DIR=/var/tmp/models/glm52-nf3-hybrid-c3b696e
MODEL_REVISION=c3b696ed315be5f8371695d308d72b0742b5d2f9

hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"
printf '%s\n' "$MODEL_REVISION" > "$MODEL_DIR/.source-revision"
```

Download once and `rsync` over the RoCE rail if that is faster, then verify the
file set and revision on every node. Do not substitute current `main`: this
recipe was validated against the older 87-shard c3 revision.

## 2. Build and distribute the image

Run on a Spark with adequate free space and no active GLM serving process:

```bash
COPY_HOSTS=192.168.192.2,192.168.192.3,192.168.192.4 \
SSH_USER=sparkuser \
./nf3-nvfp4/build-image.sh
```

The build applies three explicit corrections:

1. pin the CUDA 13.2 PyTorch/Triton/CUTLASS stack used by the validated image;
2. preserve the `12.1a` GB10 target through vLLM's CUDA-family filter; and
3. stream NF3 repacking in four-expert chunks, releasing source tensors between
   weight families to bound unified-memory peaks.

The script refuses to modify a dirty build tree. Override `BUILD_ROOT` to use a
fresh location rather than deleting existing work.

## 3. Configure, preflight, and launch

Use environment variables or a private wrapper rather than putting addresses
and key paths into the repository:

```bash
export NODES_CSV=192.168.192.1,192.168.192.2,192.168.192.3,192.168.192.4
export GIDS_CSV=3,3,3,3
export SSH_USER=sparkuser
export SSH_KEY="$HOME/.ssh/id_ed25519_cluster"
export REMOTE_HOME=/home/sparkuser
export FABRIC_IF=enp1s0f0np0
export ROCE_HCA=rocep1s0f0
export IMAGE=glm52-nf3-nvfp4:20260720
export MODEL_DIR=/var/tmp/models/glm52-nf3-hybrid-c3b696e

PROFILE=context900 ./nf3-nvfp4/launch.sh --preflight
PROFILE=context900 ./nf3-nvfp4/launch.sh --dry-run
PROFILE=context900 ./nf3-nvfp4/launch.sh
```

Workers start first and rank zero starts last. The validated launch needed about
21–22 minutes before `/health` became ready. Check status without changing the
deployment:

```bash
PROFILE=context900 ./nf3-nvfp4/launch.sh --status
HEAD_HOST=192.168.192.1 ./nf3-nvfp4/verify-runtime.sh
```

Stop only when intended:

```bash
./nf3-nvfp4/launch.sh --stop
```

## Profiles

| Profile | TP/DCP | Context | Sequences | KV/rank | Small-token collective | Intended use |
|---|---:|---:|---:|---:|---|---|
| `context900` | 4/4 | 900,000 | 4 | 8 GiB NVFP4 | B12X A2A through 64 tokens, then AG/RS | Maximum validated context |
| `fast262` | 4/1 | 262,144 | 6 | 10 GiB NVFP4 | AG/RS | Higher aggregate short-workload throughput |

Both use K=3 MTP, an 8,192-token scheduler budget, graph capture through 32,
prefix caching, and the same sparse index pattern. The 900K run reported
927,645 logical KV tokens against the checkpoint's 1,048,576-position limit.

## Measured results

All throughput entries use temperature 0, forced 512-token generations, one
warmup, three different single-stream prompts, a concurrent batch, and a
unique-nonce long prompt. “Prefill” is prompt tokens divided by full wall time
for a one-output-token request; it is not isolated kernel throughput.

| Profile | c1 median | Concurrent aggregate | Prefill approximation | Streaming decode | Median TTFT |
|---|---:|---:|---:|---:|---:|
| `context900` (c4) | 18.050 tok/s | 37.978 tok/s | 442.247 prompt tok/s over 17,945 | 18.817 tok/s | 0.5639 s |
| `fast262` (c6) | 18.912 tok/s | 48.207 tok/s | 591.300 prompt tok/s over 17,940 | 19.235 tok/s | 0.8124 s |

The 900K profile also completed an 880,007-token marker request in 2,169.407
seconds. The 45-minute owner-aware memory watch collected 87 samples per rank,
recorded zero swap, no unhealthy sample, and minimum `MemAvailable` of
5,317,700 / 7,315,812 / 7,339,108 / 7,686,500 KiB. The fast profile passed a
250,005-token marker and a 15-minute watch with the same zero-swap result.

Sanitized results and the exact measurement definitions are in
[`benchmarks/nf3-nvfp4-20260720`](../benchmarks/nf3-nvfp4-20260720/).

## Hermes compatibility

The server exposes the native `glm45` reasoning parser, `glm47` tool parser,
automatic tool choice, the checkpoint chat template, and the model name
`glm-5.2`. Configure Hermes to an OpenAI-compatible base URL ending in `/v1`
and set its context window to the selected profile.

Compatibility evidence for `context900`:

- six functional checks passed, including a native tool round trip, a real
  17.9K-token skill payload, a 4K–20K context ladder, and c4 requests;
- an exact 91-tool, 23,047-prompt-token Hermes capture returned HTTP 200 and a
  normal stop in 52.9112 seconds; and
- an end-to-end gateway request returned the expected marker.

The private request body is deliberately not included. Its body and capture
hashes are retained in the sanitized evidence so the run remains identifiable
without exposing user content or tool schemas.

## Operational cautions

- Four GB10 nodes are memory-constrained during loading, NF3 repack, JIT, and
  graph capture. A boot is not accepted merely because `/health` appears.
- Treat swap owned by a serving process as a failed stability gate. Do not run
  broad `swapoff` under pressure. Our normalization was guarded, one node at a
  time, with health checked after each node and at least 4 GiB projected free.
- The DCP4 context profile trades short-prompt and aggregate speed for logical
  context capacity. It cannot simultaneously match the DCP1 fast lane on every
  workload.
- Vendor/model-card throughput numbers use different GPU memory, batch sizes,
  prompts, sampling, and often shorter context. They are not directly
  comparable to this conservative Hermes-oriented suite.
- A clean external reproduction remains the decisive test. Please report the
  image/source pins, model revision, fabric, full vLLM command, runtime backend
  lines, and raw benchmark JSON with any result.
