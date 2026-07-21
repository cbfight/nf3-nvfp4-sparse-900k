# GLM-5.2 MXFP8/NVFP4/NF3 Hybrid with Sparse NVFP4 KV Cache at 900K Context on 4x DGX Spark

A complete, follower-replicable recipe for serving the
**`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`** checkpoint across **4x NVIDIA DGX
Spark (GB10)** nodes over a RoCE fabric, with:

- **MXFP8** non-expert projections (online quantized at load time)
- **Native mixed NVFP4/NF3** routed-expert weight execution
- **`B12X_MLA_SPARSE`** attention with GLM-5.2's full/shared indexer pattern
- **`nvfp4_ds_mla`** KV cache compression
- **TP4/DCP4** distributed serving (vLLM native multi-node, no Ray)
- **K=3 MTP** speculative decoding
- A **900,000-token served window** (validated 2026-07-20)

This is an experimental integration, not an upstream-supported vLLM configuration.

---

## What this proves

The launch is not inferred from environment variables alone. The running engine
emitted both of these selections:

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

---

## Measured results

All throughput entries use temperature 0, forced 512-token generations, one
warmup, three different single-stream prompts, a concurrent batch, and a
unique-nonce long prompt. "Prefill" is prompt tokens divided by full wall time
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
[`benchmarks/nf3-nvfp4-20260720`](benchmarks/nf3-nvfp4-20260720/).

### Hermes compatibility

The server exposes the native `glm45` reasoning parser, `glm47` tool parser,
automatic tool choice, the checkpoint chat template, and the model name
`glm-5.2`. Compatibility evidence for `context900`:

- Six functional checks passed, including a native tool round trip, a real
  17.9K-token skill payload, a 4K-20K context ladder, and c4 requests.
- An exact 91-tool, 23,047-prompt-token Hermes capture returned HTTP 200 and a
  normal stop in 52.9112 seconds.
- An end-to-end gateway request returned the expected marker.

---

## Pinned stack

| Component | Pin |
|---|---|
| Checkpoint | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` at `c3b696ed315be5f8371695d308d72b0742b5d2f9` |
| vLLM fork | `local-inference-lab/vllm` at `70c49aad686417aa2f15123731971d56edb4ded6` |
| b12x | `voipmonitor/b12x` at `1377d5f22c98de0c17d9b3f35a5b56d7587992fa` |
| Build harness | `m9e/blackwell-llm-docker` at `d3247230dfbd34b9467791c404a8ad2942c1b8e6` |
| PyTorch / Triton | `2.12.0+cu132` / `3.7.0` |
| FlashInfer | `0.6.14` |
| CUTLASS DSL | `4.5.2` |
| Validated image ID | `sha256:786af9b760056578eb57869e66d27f43ccf6fd048cf220fa0e877278468618a7` |

The image ID is evidence from our build, not a portable requirement: a clean
rebuild may have a different layer digest while retaining the pinned sources.

---

## Hardware requirements

- **4x NVIDIA DGX Spark (GB10)** -- 121 GB unified memory each, one GPU per node.
- **RoCE fabric** between the four nodes with jumbo MTU end to end.
- **Passwordless SSH** from the head node to each worker.
- At least 8 GiB `MemAvailable` per node before launch; the validated memory
  gate after startup and under load was 4 GiB.
- Roughly 350 GiB of model storage per node plus build and JIT-cache space.

Interface names and GID indices are installation-specific. Discover them with
`ip link`, `ibdev2netdev`, `ibv_devinfo`, and `show_gids`; do not copy the
example names blindly.

---

## Quick start

### 1. Download the exact checkpoint on every node

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

### 2. Build and distribute the image

Run on a Spark with adequate free space and no active GLM serving process:

```bash
COPY_HOSTS=192.168.192.2,192.168.192.3,192.168.192.4 \
SSH_USER=sparkuser \
./nf3-nvfp4/build-image.sh
```

The build applies three explicit corrections:

1. Pin the CUDA 13.2 PyTorch/Triton/CUTLASS stack used by the validated image.
2. Preserve the `12.1a` GB10 target through vLLM's CUDA-family filter.
3. Stream NF3 repacking in four-expert chunks, releasing source tensors between
   weight families to bound unified-memory peaks.

The script refuses to modify a dirty build tree. Override `BUILD_ROOT` to use
a fresh location rather than deleting existing work.

### 3. Configure, preflight, and launch

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
21-22 minutes before `/health` became ready. Check status without changing the
deployment:

```bash
PROFILE=context900 ./nf3-nvfp4/launch.sh --status
HEAD_HOST=192.168.192.1 ./nf3-nvfp4/verify-runtime.sh
```

Stop only when intended:

```bash
./nf3-nvfp4/launch.sh --stop
```

---

## Profiles

| Profile | TP/DCP | Context | Sequences | KV/rank | Small-token collective | Intended use |
|---|---:|---:|---:|---:|---|---|
| `context900` | 4/4 | 900,000 | 4 | 8 GiB NVFP4 | B12X A2A through 64 tokens, then AG/RS | Maximum validated context |
| `fast262` | 4/1 | 262,144 | 6 | 10 GiB NVFP4 | AG/RS | Higher aggregate short-workload throughput |

Both use K=3 MTP, an 8,192-token scheduler budget, graph capture through 32,
prefix caching, and the same sparse index pattern. The 900K run reported
927,645 logical KV tokens against the checkpoint's 1,048,576-position limit.

---

## Repo contents

| Path | What it is |
|---|---|
| `nf3-nvfp4/launch.sh` | Multi-node launch orchestrator (preflight, dry-run, status, stop) |
| `nf3-nvfp4/build-image.sh` | Pinned image build + distribution to worker nodes |
| `nf3-nvfp4/verify-runtime.sh` | Read-only proof that sparse MLA + NVFP4 KV are active |
| `nf3-nvfp4/sitecustomize.py` | Checkpoint adapter: dequantizes 546 non-expert MXFP8 projections to BF16, overlays MXFP8 online quant for the non-expert tier, leaves routed NVFP4/NF3 experts untouched |
| `nf3-nvfp4/README.md` | Detailed recipe and pinned source table for the NF3-NVFP4 directory |
| `patches/fix-indexer-mtp-overhang.py` | Fixes DSA indexer block-table buffer sizing under MTP (our contribution) |
| `patches/vllm-cuda13-family-archs.patch` | Preserves `12.1a` GB10 target through vLLM's CUDA-family filter |
| `patches/vllm-native-nf3-streaming-repack.patch` | Bounds NF3 repack peak on unified-memory Spark systems |
| `patches/spark-build-pins.patch` | Pins CUDA 13.2 PyTorch/Triton/CUTLASS stack in the build harness |
| `patches/spark-vllm-docker-apply-local-vllm-patch.patch` | Stages local vLLM source corrections into the Docker build |
| `patches/spark-vllm-docker-apply-native-repack-patch.patch` | Stages the NF3 streaming repack patch into the Docker build |
| `benchmarks/benchmark_openai.py` | Dependency-free client benchmark harness |
| `benchmarks/nf3-nvfp4-20260720/` | Sanitized validation record: results.json + measurement definitions |

---

## Our contributions

Things we built during this bring-up, offered back to the community:

- **NF3 streaming repack** ([`patches/vllm-native-nf3-streaming-repack.patch`](patches/vllm-native-nf3-streaming-repack.patch)):
  Bounds the native hybrid NF3 repack peak on unified-memory Spark systems by
  streaming in four-expert chunks and releasing source tensors between weight
  families. Without this, the repack OOMs on GB10's 121 GB unified memory.

- **CUDA 13.2 family-arch pin** ([`patches/vllm-cuda13-family-archs.patch`](patches/vllm-cuda13-family-archs.patch)):
  Preserves the `12.1a` GB10 target through vLLM's CUDA-family filter so the
  build does not silently drop GB10 codegen.

- **Indexer MTP-overhang fix** ([`patches/fix-indexer-mtp-overhang.py`](patches/fix-indexer-mtp-overhang.py)):
  The DSA indexer under-sizes its expanded block-table buffer by one block when
  `max_model_len` is an exact multiple of the block size and MTP is enabled.
  Crashes the engine at 3+ concurrent requests.

- **Non-expert MXFP8 checkpoint adapter** ([`nf3-nvfp4/sitecustomize.py`](nf3-nvfp4/sitecustomize.py)):
  Revision c3b696e stores 546 non-expert projections as serialized MXFP8 even
  though ModelOpt lists them under `ignore`. This adapter dequantizes exactly
  those 546 modules to BF16 at load time and overlays MXFP8 online quant for
  the non-expert tier, while leaving routed NVFP4/NF3 experts untouched.

- **Memory-budget numbers** for exactly 900K context on this checkpoint: gmu
  0.88 + 8 GiB NVFP4 KV per rank for the context profile, 10 GiB for the fast
  profile.

- **Load-phase page-cache-drop procedure**: periodic `drop_caches` on every
  node during weight load unsticks GB10 kernel-reclaim stalls.

---

## Contributors

- **[cbfight (Wesley Wong)](https://github.com/cbfight)** — NF3 streaming repack, CUDA 13.2 family-arch pin, indexer MTP-overhang fix, MXFP8 checkpoint adapter, 900K memory-budget validation, recipe author.
- **[TonyD2wild](https://github.com/TonyD2wild)** — Original 200K QuantTrio recipe, predecessor build harness, and foundational cluster configuration that this NF3 recipe builds upon.

---

## Credits and upstream lineage

This recipe stands on the shoulders of the people and projects below. If you
use it, their work is what you are using.

| Who | What |
|---|---|
| **[CosmicRaisins](https://github.com/CosmicRaisins/glm-5.2-gb10)** | The whole sm_121 sparse-MLA port: the `glm-5.2-gb10` repo, the 10 Triton kernels, the DeepGEMM bypass, the launch harness pattern. Apache-2.0. |
| **[madeby561](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)** | The MXFP8/NVFP4/NF3 hybrid checkpoint itself. MIT. |
| **[local-inference-lab](https://github.com/local-inference-lab/vllm)** | The vLLM fork with native NF3/NVFP4 hybrid support. Apache-2.0. |
| **[voipmonitor/b12x](https://github.com/voipmonitor/b12x)** | CUDA-graph-safe decode kernel for GLM-5.2 sparse MLA. |
| **[m9e/blackwell-llm-docker](https://github.com/m9e/blackwell-llm-docker)** | The image build harness adapted for this recipe. |
| **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** | The original image build harness pattern for GB10. |
| **Zatz** | Unpruned QuantTrio recipe proving the full 256-expert checkpoint fits on 4x GB10 (NVIDIA forum thread [374125](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125), posts #57/#84). |
| **back199640** | Tuning that closed the gap: `--async-scheduling`, MTP k=4 with `draft_tensor_parallel_size: 1`, head-pad trick, explicit `--kv-cache-memory-bytes` (thread 374125, posts #80/#89). |
| **ciprianveg** | NCCL channel-narrowing find (`NCCL_MIN/MAX_NCHANNELS=4`, thread 374125 post #107). |
| **p33zy** | Explored the alternative NVFP4 quantization path and GB10 hardware-acceleration trade-offs (thread 374125). |
| **[QuantTrio](https://huggingface.co/QuantTrio)** | The `GLM-5.2-Int4-Int8Mix` checkpoint (used by the predecessor 200K recipe, not by this NF3 recipe). |

Forum threads (read both -- they are the primary sources):

- https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125
- https://forums.developer.nvidia.com/t/followup-mystery-solved-4x-spark-glm-5-2-nfp4-24tp-s-128k-ctx-no-reap/375416

---

## Operational cautions

- Four GB10 nodes are memory-constrained during loading, NF3 repack, JIT, and
  graph capture. A boot is not accepted merely because `/health` appears.
- Treat swap owned by a serving process as a failed stability gate. Do not run
  broad `swapoff` under pressure. Our normalization was guarded, one node at a
  time, with health checked after each node and at least 4 GiB projected free.
- The DCP4 context profile trades short-prompt and aggregate speed for logical
  context capacity. It cannot simultaneously match the DCP1 fast lane on
  every workload.
- Vendor/model-card throughput numbers use different GPU memory, batch sizes,
  prompts, sampling, and often shorter context. They are not directly
  comparable to this conservative Hermes-oriented suite.
- A clean external reproduction remains the decisive test. Please report the
  image/source pins, model revision, fabric, full vLLM command, runtime
  backend lines, and raw benchmark JSON with any result.

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
