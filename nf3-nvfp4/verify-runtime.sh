#!/usr/bin/env bash
# Read-only proof that sparse MLA and NVFP4 MLA KV are active together.
set -euo pipefail

HEAD_HOST="${HEAD_HOST:-192.168.192.1}"
SSH_USER="${SSH_USER:-sparkuser}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_cluster}"
CONTAINER_NAME="${CONTAINER_NAME:-glm52_nf3_nvfp4}"
PORT="${PORT:-8210}"

ssh_opts=(-i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

curl -fsS "http://$HEAD_HOST:$PORT/health" >/dev/null
curl -fsS "http://$HEAD_HOST:$PORT/v1/models" | python3 -m json.tool

ssh "${ssh_opts[@]}" "$SSH_USER@$HEAD_HOST" bash -s -- "$CONTAINER_NAME" <<'REMOTE'
set -euo pipefail
name="$1"
docker inspect -f 'running={{.State.Running}} oom={{.State.OOMKilled}} exit={{.State.ExitCode}} restarts={{.RestartCount}} image={{.Config.Image}}' "$name"
env_text="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$name")"
printf '%s\n' "$env_text" | grep -E '^(VLLM_USE_B12X_DCP_A2A|VLLM_DCP_A2A_MAX_TOKENS|VLLM_NF3_PACK_CHUNK)='
cmd_text="$(docker inspect -f '{{join .Config.Cmd " "}}' "$name")"
printf '%s\n' "$cmd_text" | grep -F -- '--quantization nvfp4_nf3_hybrid'
printf '%s\n' "$cmd_text" | grep -F -- '--kv-cache-dtype nvfp4_ds_mla'
printf '%s\n' "$cmd_text" | grep -F -- '--attention-backend B12X_MLA_SPARSE'
logs="$(docker logs "$name" 2>&1)"
printf '%s\n' "$logs" | grep -F 'Using AttentionBackendEnum.B12X_MLA_SPARSE backend.' | tail -1
printf '%s\n' "$logs" | grep -F 'Using nvfp4_ds_mla data type to store kv cache.' | tail -1
REMOTE

echo 'verification=passed sparse_attention=yes nvfp4_kv=yes'
