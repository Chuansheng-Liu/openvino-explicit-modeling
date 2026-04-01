#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENVINO_DIR="${ROOT_DIR}/openvino"
GENAI_DIR="${ROOT_DIR}/openvino.genai"
OPENVINO_URL="https://github.com/liangali/openvino.git"
GENAI_URL="https://github.com/liangali/openvino.genai"
TARGET_BRANCH="explicit-modeling"
PATCH_SCRIPT="${SCRIPT_DIR}/scripts/apply_onednn_patch.sh"

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

ensure_git() {
    if ! command -v git >/dev/null 2>&1; then
        log_error "git not found in PATH."
        return 1
    fi
}

checkout_branch() {
    local repo_dir="$1"
    local branch="$2"

    if git -C "${repo_dir}" show-ref --verify --quiet "refs/heads/${branch}"; then
        git -C "${repo_dir}" checkout "${branch}"
        return 0
    fi

    if git -C "${repo_dir}" show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
        git -C "${repo_dir}" checkout -b "${branch}" "origin/${branch}"
        return 0
    fi

    log_error "Failed to find branch ${branch} in ${repo_dir}."
    return 1
}

ensure_repo() {
    local repo_dir="$1"
    local repo_url="$2"
    local repo_branch="$3"
    local repo_name="$4"

    if [[ -d "${repo_dir}/.git" ]]; then
        log_info "${repo_name} already exists: ${repo_dir}"
    else
        if [[ -e "${repo_dir}" ]]; then
            log_error "${repo_name} path exists but is not a git repo: ${repo_dir}"
            return 1
        fi

        echo "[CLONE] ${repo_name}"
        git clone "${repo_url}" "${repo_dir}"
    fi

    echo "[CHECKOUT] ${repo_name} -> ${repo_branch}"
    git -C "${repo_dir}" fetch --all --tags
    checkout_branch "${repo_dir}" "${repo_branch}"

    echo "[SUBMODULE] ${repo_name}"
    git -C "${repo_dir}" submodule update --init --recursive
}

main() {
    log_info "repo.sh path      : ${SCRIPT_DIR}"
    log_info "workspace root    : ${ROOT_DIR}"

    ensure_git
    ensure_repo "${OPENVINO_DIR}" "${OPENVINO_URL}" "${TARGET_BRANCH}" "openvino"
    ensure_repo "${GENAI_DIR}" "${GENAI_URL}" "${TARGET_BRANCH}" "openvino.genai"

    if [[ ! -f "${PATCH_SCRIPT}" ]]; then
        log_error "Patch helper not found: ${PATCH_SCRIPT}"
        return 1
    fi

    echo "[PATCH] onednn_gpu"
    "${PATCH_SCRIPT}"

    echo "[OK] Repository setup finished."
    echo "     Ready to build from: ${ROOT_DIR}"
}

main "$@"