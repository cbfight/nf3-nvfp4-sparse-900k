#!/usr/bin/env bash
# Launch the validated GLM-5.2 native NF3 + sparse NVFP4-KV profiles.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Override these in the environment or a private wrapper. Do not commit private
# hostnames, keys, or addresses to this repository.
NODES_CSV="${NODES_CSV:-192.168.192.1,192.168.192.2,192.168.192.3,192.168.192.4}"
GIDS_CSV="${GIDS_CSV:-3,3,3,3}"
SSH_USER="${SSH_USER:-sparkuser}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_cluster}"
REMOTE_HOME="${REMOTE_HOME:-/home/$SSH_USER}"
FABRIC_IF="${FABRIC_IF:-enp1s0f0np0}"
ROCE_HCA="${ROCE_HCA:-rocep1s0f0}"
IMAGE="${IMAGE:-glm52-nf3-nvfp4:20260720}"
MODEL_DIR="${MODEL_DIR:-/var/tmp/models/glm52-nf3-hybrid-c3b696e}"
MODEL_REVISION="${MODEL_REVISION:-c3b696ed315be5f8371695d308d72b0742b5d2f9}"
SUPPORT_DIR="${SUPPORT_DIR:-$REMOTE_HOME/.local/share/glm52-nf3-nvfp4}"
NAME="${NAME:-glm52_nf3_nvfp4}"
PORT="${PORT:-8210}"
MASTER_PORT="${MASTER_PORT:-29501}"
PROFILE="${PROFILE:-context900}"

IFS=',' read -r -a NODES <<<"$NODES_CSV"
IFS=',' read -r -a GIDS <<<"$GIDS_CSV"
HEAD="${NODES[0]:-}"
NNODES="${#NODES[@]}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
say() { printf '\n== %s ==\n' "$*"; }

[ "$NNODES" -eq 4 ] || die "this recipe was validated on exactly four nodes"
[ "${#GIDS[@]}" -eq "$NNODES" ] || die "GIDS_CSV must have one entry per node"

ACTION=launch
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --preflight) ACTION=preflight ;;
    --status) ACTION=status ;;
    --stop) ACTION=stop ;;
    *) die "unknown argument: $arg" ;;
  esac
done

case "$PROFILE" in
  context900)
    MAX_MODEL_LEN=900000
    MAX_NUM_SEQS=4
    DCP=4
    KV_BYTES=8589934592
    USE_A2A=1
    COMPILATION_CONFIG='{"pass_config":{"fuse_allreduce_rms":true}}'
    ;;
  fast262)
    MAX_MODEL_LEN=262144
    MAX_NUM_SEQS=6
    DCP=1
    KV_BYTES=10737418240
    USE_A2A=0
    COMPILATION_CONFIG=''
    ;;
  *) die "PROFILE must be context900 or fast262" ;;
esac

ssh_opts=(-i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

remote() {
  local node="$1"; shift
  ssh "${ssh_opts[@]}" "$SSH_USER@$node" "$@"
}

if [ "$ACTION" = stop ]; then
  for node in "${NODES[@]}"; do
    remote "$node" "docker rm -f $(printf %q "$NAME") 2>/dev/null || true"
  done
  exit 0
fi

if [ "$ACTION" = status ]; then
  for node in "${NODES[@]}"; do
    printf '%s ' "$node"
    remote "$node" "docker inspect -f 'running={{.State.Running}} oom={{.State.OOMKilled}} exit={{.State.ExitCode}} restarts={{.RestartCount}} image={{.Config.Image}}' $(printf %q "$NAME") 2>/dev/null || echo absent"
  done
  curl -fsS --max-time 5 "http://$HEAD:$PORT/health" >/dev/null \
    && echo "endpoint=healthy url=http://$HEAD:$PORT/v1" \
    || echo "endpoint=not-ready"
  exit 0
fi

preflight_node='set -euo pipefail
[ "$(cat /sys/class/net/$FABRIC_IF/mtu)" -ge 9000 ]
[ "$(cat /sys/class/net/$FABRIC_IF/carrier)" = 1 ]
ibv_devinfo -d "$ROCE_HCA" 2>/dev/null | grep -Eq "state:[[:space:]]+PORT_ACTIVE"
docker image inspect "$IMAGE" >/dev/null
[ -d "$MODEL_DIR" ]
[ "$(cat "$MODEL_DIR/.source-revision")" = "$MODEL_REVISION" ]
mem_kb="$(awk '\''/^MemAvailable:/ {print $2}'\'' /proc/meminfo)"
[ "$mem_kb" -ge 8388608 ]
printf "host=%s mtu=%s mem_available_kb=%s image=%s" "$(hostname)" "$(cat /sys/class/net/$FABRIC_IF/mtu)" "$mem_kb" "$(docker image inspect -f "{{.Id}}" "$IMAGE")"'

run_preflight() {
  local rank node
  local ids=''
  for rank in "${!NODES[@]}"; do
    node="${NODES[$rank]}"
    printf '%s ' "$node"
    result="$(remote "$node" env \
      "FABRIC_IF=$FABRIC_IF" "ROCE_HCA=$ROCE_HCA" "IMAGE=$IMAGE" \
      "MODEL_DIR=$MODEL_DIR" "MODEL_REVISION=$MODEL_REVISION" \
      bash -c "$(printf %q "$preflight_node")")"
    echo "$result"
    ids+=" ${result##*image=}"
  done
  [ "$(printf '%s\n' $ids | sort -u | wc -l | tr -d ' ')" -eq 1 ] \
    || die "image IDs differ across nodes"
}

if [ "$ACTION" = preflight ]; then
  run_preflight
  exit 0
fi

if [ "$DRY_RUN" = 0 ]; then
  run_preflight

  # The adapter is small and is staged explicitly on every node. Model weights,
  # the image, and runtime caches remain node-local.
  for node in "${NODES[@]}"; do
    remote "$node" "mkdir -p $(printf %q "$SUPPORT_DIR")"
    scp "${ssh_opts[@]}" "$SCRIPT_DIR/sitecustomize.py" \
      "$SSH_USER@$node:$SUPPORT_DIR/sitecustomize.py" >/dev/null
  done
else
  say "dry run; preflight and adapter staging skipped"
fi

SPARSE_PATTERN='FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS'
HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"$SPARSE_PATTERN\"}"
SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'

docker_cmd() {
  local rank="$1" gid="$2" headless="$3"
  local cmd=(docker run -d --name "$NAME"
    --privileged --security-opt label=disable --network host --ipc host --gpus all
    -v "$MODEL_DIR:$MODEL_DIR:ro"
    -v "$SUPPORT_DIR/sitecustomize.py:/opt/glm52-adapter/sitecustomize.py:ro"
    -v "$REMOTE_HOME/.cache/huggingface:/root/.cache/huggingface"
    -v "$REMOTE_HOME/.cache/vllm:/root/.cache/vllm"
    -v "$REMOTE_HOME/.cache/flashinfer:/root/.cache/flashinfer"
    -v "$REMOTE_HOME/.triton:/root/.triton"
    -e "PYTHONPATH=/opt/glm52-adapter"
    -e "VLLM_HOST_IP=${NODES[$rank]}"
    -e "NCCL_SOCKET_IFNAME=$FABRIC_IF" -e "GLOO_SOCKET_IFNAME=$FABRIC_IF"
    -e "TP_SOCKET_IFNAME=$FABRIC_IF" -e "UCX_NET_DEVICES=$FABRIC_IF"
    -e "OMPI_MCA_btl_tcp_if_include=$FABRIC_IF" -e "MN_IF_NAME=$FABRIC_IF"
    -e "NCCL_IB_HCA=$ROCE_HCA" -e "NCCL_IB_GID_INDEX=$gid"
    -e "NCCL_IB_DISABLE=0" -e "NCCL_IGNORE_CPU_AFFINITY=1" -e "NCCL_DEBUG=WARN"
    -e "CUTE_DSL_ARCH=sm_121a" -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600"
    -e "VLLM_USE_V2_MODEL_RUNNER=1" -e "VLLM_USE_B12X_FP8_GEMM=1"
    -e "VLLM_USE_B12X_MOE=1" -e "VLLM_USE_B12X_SPARSE_INDEXER=1"
    -e "VLLM_DCP_GLOBAL_TOPK=1" -e "VLLM_DCP_SHARD_DRAFT=1"
    -e "VLLM_USE_B12X_DCP_A2A=$USE_A2A" -e "VLLM_DCP_A2A_MAX_TOKENS=64"
    -e "VLLM_DCP_A2A_LARGE_BACKEND=ag_rs"
    -e "VLLM_NF3_PACK_CHUNK=4" -e "VLLM_NF3_REPACK_EMPTY_CACHE=1"
    -e "B12X_W4A16_TC_DECODE=1" -e "B12X_MOE_FORCE_A16=1"
    -e "B12X_DENSE_SPLITK_TURBO=1" -e "HYBRID_TIER=both"
    -e "HYBRID_KEPT=b12x_nf3" -e "HYBRID_NF3=b12x_nf3"
    -e "HYBRID_B12X_MAX_TOKENS=8192" -e "HYBRID_MXFP8_NATIVE=1"
    -e "HYBRID_MXFP8_TIER_JSON=/usr/local/lib/python3.12/dist-packages/mxfp8_tier.json"
    -e "VLLM_DO_NOT_TRACK=1" -e "DO_NOT_TRACK=1"
    "$IMAGE"
    vllm serve "$MODEL_DIR"
    --served-model-name glm-5.2 --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size 4 --decode-context-parallel-size "$DCP"
    --dcp-comm-backend ag_rs --dcp-kv-cache-interleave-size 1
    --quantization nvfp4_nf3_hybrid --kv-cache-dtype nvfp4_ds_mla
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens 8192 --max-cudagraph-capture-size 32
    --gpu-memory-utilization 0.88 --kv-cache-memory-bytes "$KV_BYTES"
    --attention-backend B12X_MLA_SPARSE --moe-backend b12x
    --load-format safetensors --async-scheduling --enable-chunked-prefill
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
    --speculative-config "$SPECULATIVE_CONFIG" --hf-overrides "$HF_OVERRIDES"
    --enable-prefix-caching --chat-template "$MODEL_DIR/chat_template.jinja"
    --chat-template-content-format string
    --default-chat-template-kwargs '{"reasoning_effort":"high"}'
    --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD"
    --master-port "$MASTER_PORT")
  [ -n "$COMPILATION_CONFIG" ] && cmd+=(--compilation-config "$COMPILATION_CONFIG")
  [ "$headless" = 1 ] && cmd+=(--headless)
  printf '%q ' "${cmd[@]}"
}

say "launching profile=$PROFILE head=$HEAD:$PORT"
for ((rank=1; rank<NNODES; rank++)); do
  command="docker rm -f $(printf %q "$NAME") 2>/dev/null || true; $(docker_cmd "$rank" "${GIDS[$rank]}" 1)"
  if [ "$DRY_RUN" = 1 ]; then
    printf 'ssh %q@%q %q\n' "$SSH_USER" "${NODES[$rank]}" "$command"
  else
    remote "${NODES[$rank]}" "$command"
  fi
done

command="docker rm -f $(printf %q "$NAME") 2>/dev/null || true; $(docker_cmd 0 "${GIDS[0]}" 0)"
if [ "$DRY_RUN" = 1 ]; then
  printf 'ssh %q@%q %q\n' "$SSH_USER" "$HEAD" "$command"
else
  remote "$HEAD" "$command"
  say "started; expect roughly 20-25 minutes before health is ready"
  printf 'poll: curl -fsS http://%s:%s/health\n' "$HEAD" "$PORT"
fi
