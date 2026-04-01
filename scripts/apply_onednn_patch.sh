#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PATCH_FILE="${SCRIPT_DIR}/../patches/onednn_gpu.patch"
ONEDNN_GPU_DIR="${ROOT_DIR}/openvino/src/plugins/intel_gpu/thirdparty/onednn_gpu"

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "[ERROR] Patch file not found: ${PATCH_FILE}" >&2
    exit 1
fi

if [[ ! -d "${ONEDNN_GPU_DIR}" ]]; then
    echo "[ERROR] onednn_gpu directory not found: ${ONEDNN_GPU_DIR}" >&2
    exit 1
fi

if git -C "${ONEDNN_GPU_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    git -C "${ONEDNN_GPU_DIR}" apply "${PATCH_FILE}"
    echo "[apply_onednn_patch] Patch applied successfully."
elif git -C "${ONEDNN_GPU_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "[apply_onednn_patch] Patch already applied, skipping."
else
    echo "[ERROR] onednn_gpu.patch does not apply. Check onednn_gpu version." >&2
    exit 1
fi