#!/bin/bash
# Build the ov_serve Docker image.
# Usage: ./build_docker.sh [build_dir] [image_name]
#   build_dir:  directory containing Release/ (default: /deploy/qwen3_5_9b_g128)
#   image_name: docker image name (default: derived from build_dir basename)

set -e
BUILD_DIR="${1:-/deploy/qwen3_5_9b_g128}"
IMAGE_NAME="${2:-ov_serve_$(basename "$BUILD_DIR"):latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVERS_DIR="/home/gta/chuansheng/drivers"

# Ensure drivers/ exists in build dir
if [ ! -d "$BUILD_DIR/drivers" ] && [ -d "$DRIVERS_DIR" ]; then
    echo "Copying $DRIVERS_DIR -> $BUILD_DIR/drivers"
    cp -r "$DRIVERS_DIR" "$BUILD_DIR/drivers"
fi

cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"

cd "$BUILD_DIR"
docker build -t "$IMAGE_NAME" .
echo "Done. Run with:"
echo "  docker run --rm --device /dev/dri --group-add video -p 8080:8080 $IMAGE_NAME"
