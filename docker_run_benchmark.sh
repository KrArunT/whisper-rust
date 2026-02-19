#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-whisper-rust-bench:latest}"
CPUSET_CPUS="${CPUSET_CPUS:-0-7}"

DOCKER_BUILDKIT=1 docker build \
  --cache-from "$IMAGE_NAME" \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  -t "$IMAGE_NAME" \
  .

docker run --rm -it \
  --cpuset-cpus="$CPUSET_CPUS" \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$IMAGE_NAME" \
  ./discover_optimal_model.sh "$@"
