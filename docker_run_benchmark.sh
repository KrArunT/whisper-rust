#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-whisper-rust-bench:latest}"
CPUSET_CPUS="${CPUSET_CPUS:-0-7}"

CACHE_FROM_ARGS=()
if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  CACHE_FROM_ARGS+=(--cache-from "$IMAGE_NAME")
fi

DOCKER_BUILDKIT=1 docker build \
  "${CACHE_FROM_ARGS[@]}" \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  -t "$IMAGE_NAME" \
  .

docker run --rm -it \
  --cpuset-cpus="$CPUSET_CPUS" \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  "$@"
