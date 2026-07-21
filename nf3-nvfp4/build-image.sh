#!/usr/bin/env bash
# Build the pinned native NF3 + NVFP4-KV runtime used by this recipe.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

BUILD_REPO="${BUILD_REPO:-https://github.com/m9e/blackwell-llm-docker.git}"
BUILD_COMMIT="${BUILD_COMMIT:-d3247230dfbd34b9467791c404a8ad2942c1b8e6}"
VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-70c49aad686417aa2f15123731971d56edb4ded6}"
B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
B12X_COMMIT="${B12X_COMMIT:-1377d5f22c98de0c17d9b3f35a5b56d7587992fa}"
IMAGE_TAG="${IMAGE_TAG:-glm52-nf3-nvfp4:20260720}"
BUILD_ROOT="${BUILD_ROOT:-$REPO_ROOT/.build/blackwell-llm-docker-nf3}"
BUILD_JOBS="${BUILD_JOBS:-8}"
COPY_HOSTS="${COPY_HOSTS:-}"
SSH_USER="${SSH_USER:-$USER}"

if [ ! -d "$BUILD_ROOT/.git" ]; then
  mkdir -p "$(dirname -- "$BUILD_ROOT")"
  git clone "$BUILD_REPO" "$BUILD_ROOT"
fi

if [ -n "$(git -C "$BUILD_ROOT" status --porcelain)" ]; then
  printf 'build tree is not clean: %s\n' "$BUILD_ROOT" >&2
  printf 'choose a fresh BUILD_ROOT; this script will not discard existing work\n' >&2
  exit 2
fi
git -C "$BUILD_ROOT" fetch --depth=1 origin "$BUILD_COMMIT"
git -C "$BUILD_ROOT" checkout --detach "$BUILD_COMMIT"

git -C "$BUILD_ROOT" apply --check "$REPO_ROOT/patches/spark-build-pins.patch"
git -C "$BUILD_ROOT" apply "$REPO_ROOT/patches/spark-build-pins.patch"

cp "$REPO_ROOT/patches/vllm-cuda13-family-archs.patch" "$BUILD_ROOT/"
cp "$REPO_ROOT/patches/vllm-native-nf3-streaming-repack.patch" "$BUILD_ROOT/"

git -C "$BUILD_ROOT" apply --check \
  "$REPO_ROOT/patches/spark-vllm-docker-apply-local-vllm-patch.patch"
git -C "$BUILD_ROOT" apply \
  "$REPO_ROOT/patches/spark-vllm-docker-apply-local-vllm-patch.patch"
git -C "$BUILD_ROOT" apply --check \
  "$REPO_ROOT/patches/spark-vllm-docker-apply-native-repack-patch.patch"
git -C "$BUILD_ROOT" apply \
  "$REPO_ROOT/patches/spark-vllm-docker-apply-native-repack-patch.patch"

git -C "$BUILD_ROOT" diff --check
sha256sum \
  "$BUILD_ROOT/Dockerfile" \
  "$BUILD_ROOT/vllm-cuda13-family-archs.patch" \
  "$BUILD_ROOT/vllm-native-nf3-streaming-repack.patch"

args=(
  --gpu-arch 12.1a
  -j "$BUILD_JOBS"
  --vllm-repo "$VLLM_REPO"
  --vllm-ref "$VLLM_COMMIT"
  --vllm-commit "$VLLM_COMMIT"
  --b12x-repo "$B12X_REPO"
  --b12x-ref "$B12X_COMMIT"
  --b12x-commit "$B12X_COMMIT"
  --full-log
  -t "$IMAGE_TAG"
)

if [ -n "$COPY_HOSTS" ]; then
  args+=(--copy-to "$COPY_HOSTS" --copy-parallel --user "$SSH_USER")
fi

printf 'Building %s\n' "$IMAGE_TAG"
printf 'vLLM=%s b12x=%s build-harness=%s\n' \
  "$VLLM_COMMIT" "$B12X_COMMIT" "$BUILD_COMMIT"
cd "$BUILD_ROOT"
exec ./build-and-copy.sh "${args[@]}"
