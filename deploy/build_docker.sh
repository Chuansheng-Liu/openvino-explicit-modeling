#!/bin/bash
# Build the ov_serve Docker image.
# Usage: ./build_docker.sh [build_dir]
#   build_dir: directory containing Release/ and drivers/ (default: /deploy/qwen3_5_9b_g32)

set -e
BUILD_DIR="${1:-/deploy/qwen3_5_9b_g32}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVERS_DIR="/home/gta/chuansheng/drivers"

# Ensure drivers/ exists in build dir
if [ ! -d "$BUILD_DIR/drivers" ] && [ -d "$DRIVERS_DIR" ]; then
    echo "Linking $DRIVERS_DIR -> $BUILD_DIR/drivers"
    ln -sfn "$DRIVERS_DIR" "$BUILD_DIR/drivers"
fi

cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"

cd "$BUILD_DIR"
docker build -t qwen3_5_9b_ov_serve_g32_prefix_cache:latest .
echo "Done. Run with:"
echo "  docker run --rm --device /dev/dri --group-add video -p 8080:8080 qwen3_5_9b_ov_serve_g32_prefix_cache:latest"
