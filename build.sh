#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENVINO_SRC="${ROOT}/openvino"
OPENVINO_BUILD="${OPENVINO_SRC}/build"
GENAI_SRC="${ROOT}/openvino.genai"
GENAI_BUILD="${GENAI_SRC}/build"
GENAI_BIN_ROOT="${GENAI_BUILD}/bin"
WHEEL_OUTPUT_ROOT="${ROOT}/wheel"
WHEEL_VENV_ROOT="${SCRIPT_DIR}/.wheel-build-venv"
WHEEL_SCRIPT="${SCRIPT_DIR}/scripts/wheel.py"

BUILD_OPENVINO=1
BUILD_GENAI=1
BUILD_WHEEL=0
BUILD_TYPE="${BUILD_TYPE:-Release}"
WHEEL_PYTHON_REQUEST=""

WHEEL_OUTPUT_DIR=""
WHEEL_VENV_DIR=""
WHEEL_PYTHON=""
WHEEL_PYTHON_SOURCE=""
WHEEL_PYTHON_VERSION=""
WHEEL_TAG=""
HOST_PYTHON=""
OPENVINO_WHEEL_PATH=""

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

usage() {
    cat <<EOF
Usage:
  build.sh
      Configure and build openvino, then configure and build openvino.genai.

  build.sh --wheel
  build.sh --wheel --python=3.11.9
      Build openvino, build openvino.genai, and create Python wheel files in:
      the wheel/cpXY folder under the workspace root.
      The Python version defaults to the first python3/python found in PATH unless --python is specified.
      Wheel build virtual environments are reused under .wheel-build-venv/<python-version>.

  build.sh --help
      Show this help message.
EOF
}

ensure_command() {
    local command_name="$1"
    local hint="$2"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        log_error "${command_name} not found in PATH."
        if [[ -n "${hint}" ]]; then
            echo "        ${hint}" >&2
        fi
        return 1
    fi
}

ensure_build_tools() {
    ensure_command cmake "Install CMake 3.x and make sure it is available in PATH."
    ensure_command ninja "Install Ninja and make sure it is available in PATH."
}

ensure_uv() {
    ensure_command uv "Install uv and make sure it is available in PATH before running build.sh --wheel."
}

ensure_host_python() {
    if [[ -n "${HOST_PYTHON}" ]]; then
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        HOST_PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        HOST_PYTHON="$(command -v python)"
    else
        log_error "python3/python not found in PATH."
        echo "        Install Python 3.10+ and make sure it is available in PATH." >&2
        return 1
    fi
}

get_python_version() {
    local python_exe="$1"
    "${python_exe}" -c 'import sys; print(".".join(str(x) for x in sys.version_info[:3]))'
}

make_wheel_tag() {
    local version="$1"
    local major minor _rest
    IFS=. read -r major minor _rest <<< "${version}"
    if [[ -z "${major}" || -z "${minor}" ]]; then
        return 1
    fi
    printf 'cp%s%s' "${major}" "${minor}"
}

resolve_wheel_context() {
    if [[ -n "${WHEEL_CONTEXT_READY:-}" ]]; then
        return 0
    fi

    ensure_uv

    if [[ -n "${WHEEL_PYTHON_REQUEST}" ]]; then
        echo "[SETUP] Ensuring Python ${WHEEL_PYTHON_REQUEST} is available via uv"
        uv python install "${WHEEL_PYTHON_REQUEST}"
        WHEEL_PYTHON_SOURCE="$(uv python find "${WHEEL_PYTHON_REQUEST}")"
        if [[ -z "${WHEEL_PYTHON_SOURCE}" ]]; then
            log_error "uv could not resolve a managed Python interpreter for ${WHEEL_PYTHON_REQUEST}."
            return 1
        fi
        WHEEL_PYTHON_VERSION="$(get_python_version "${WHEEL_PYTHON_SOURCE}")"
    else
        ensure_host_python
        WHEEL_PYTHON_SOURCE="${HOST_PYTHON}"
        WHEEL_PYTHON_VERSION="$(get_python_version "${HOST_PYTHON}")"
    fi

    WHEEL_TAG="$(make_wheel_tag "${WHEEL_PYTHON_VERSION}")"
    WHEEL_VENV_DIR="${WHEEL_VENV_ROOT}/${WHEEL_PYTHON_VERSION}"
    WHEEL_OUTPUT_DIR="${WHEEL_OUTPUT_ROOT}/${WHEEL_TAG}"

    log_info "Wheel build Python: ${WHEEL_PYTHON_VERSION} (${WHEEL_TAG})"
    if [[ -n "${WHEEL_PYTHON_REQUEST}" ]]; then
        log_info "Resolved from --python=${WHEEL_PYTHON_REQUEST}"
    else
        log_info "Using default python from PATH: ${WHEEL_PYTHON_SOURCE}"
    fi

    WHEEL_CONTEXT_READY=1
}

ensure_wheel_python() {
    if [[ -n "${WHEEL_ENV_READY:-}" ]]; then
        return 0
    fi

    resolve_wheel_context

    local wheel_venv_info_file create_wheel_venv current_version current_source
    wheel_venv_info_file="${WHEEL_VENV_DIR}/.wheel-env-info.txt"
    WHEEL_PYTHON="${WHEEL_VENV_DIR}/bin/python"

    mkdir -p "${WHEEL_VENV_ROOT}"

    create_wheel_venv=0
    if [[ -x "${WHEEL_PYTHON}" ]]; then
        if [[ ! -f "${wheel_venv_info_file}" ]]; then
            echo "[SETUP] Existing wheel build venv is missing metadata and will be recreated: ${WHEEL_VENV_DIR}"
            create_wheel_venv=1
        else
            IFS='|' read -r current_version current_source < "${wheel_venv_info_file}" || true
            if [[ -z "${current_version}" || -z "${current_source}" ]]; then
                echo "[SETUP] Existing wheel build venv metadata is invalid and will be recreated: ${WHEEL_VENV_DIR}"
                create_wheel_venv=1
            elif [[ "${current_version}" != "${WHEEL_PYTHON_VERSION}" ]]; then
                echo "[SETUP] Existing wheel build venv uses Python ${current_version} and will be recreated for ${WHEEL_PYTHON_VERSION}."
                create_wheel_venv=1
            elif [[ "${current_source}" != "${WHEEL_PYTHON_SOURCE}" ]]; then
                echo "[SETUP] Existing wheel build venv uses ${current_source} and will be recreated for ${WHEEL_PYTHON_SOURCE}."
                create_wheel_venv=1
            else
                echo "[SETUP] Reusing wheel build venv: ${WHEEL_VENV_DIR}"
            fi
        fi
    else
        create_wheel_venv=1
    fi

    if [[ "${create_wheel_venv}" == "1" ]]; then
        rm -rf "${WHEEL_VENV_DIR}"
        echo "[SETUP] Creating wheel build venv for Python ${WHEEL_PYTHON_VERSION}: ${WHEEL_VENV_DIR}"
        uv venv --seed --python "${WHEEL_PYTHON_SOURCE}" "${WHEEL_VENV_DIR}"
        printf '%s|%s\n' "${WHEEL_PYTHON_VERSION}" "${WHEEL_PYTHON_SOURCE}" > "${wheel_venv_info_file}"
    fi

    echo "[SETUP] Installing wheel build dependencies"
    "${WHEEL_PYTHON}" -m pip install --upgrade pip
    "${WHEEL_PYTHON}" -m pip install --upgrade 'setuptools>=70.1' wheel build packaging 'py-build-cmake==0.5.0' 'pybind11-stubgen==2.5.5'

    WHEEL_ENV_READY=1
}

prepare_wheel_output() {
    mkdir -p "${WHEEL_OUTPUT_DIR}"
    rm -f "${WHEEL_OUTPUT_DIR}"/openvino-*.whl
    rm -f "${WHEEL_OUTPUT_DIR}"/openvino_genai-*.whl
    rm -f "${WHEEL_OUTPUT_DIR}"/openvino_tokenizers-*.whl
    rm -f "${WHEEL_OUTPUT_DIR}"/numpy-*.whl
    rm -f "${WHEEL_OUTPUT_DIR}"/openvino_telemetry-*.whl
    rm -f "${WHEEL_OUTPUT_DIR}/wheel.py"
}

copy_latest_openvino_wheel() {
    shopt -s nullglob
    local wheels=("${OPENVINO_BUILD}"/wheels/openvino-*-${WHEEL_TAG}-${WHEEL_TAG}-*.whl)
    shopt -u nullglob

    if [[ ${#wheels[@]} -eq 0 ]]; then
        log_error "Failed to locate the newly built openvino wheel for ${WHEEL_TAG} in ${OPENVINO_BUILD}/wheels."
        return 1
    fi

    OPENVINO_WHEEL_PATH="${wheels[0]}"
    cp -f "${OPENVINO_WHEEL_PATH}" "${WHEEL_OUTPUT_DIR}/"
    OPENVINO_WHEEL_PATH="${WHEEL_OUTPUT_DIR}/$(basename "${OPENVINO_WHEEL_PATH}")"
}

configure_openvino() {
    if [[ -n "${OPENVINO_ALREADY_CONFIGURED:-}" ]]; then
        return 0
    fi

    local cmake_args
    cmake_args=("-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
    if [[ "${BUILD_WHEEL}" == "1" ]]; then
        ensure_wheel_python
        cmake_args+=("-DENABLE_PYTHON=ON" "-DENABLE_WHEEL=ON" "-DPython3_EXECUTABLE=${WHEEL_PYTHON}")
    else
        cmake_args+=(
            "-DENABLE_PYTHON=OFF"
            "-DENABLE_WHEEL=OFF"
            "-UPython3_EXECUTABLE"
            "-UPython3_ROOT_DIR"
            "-UPython3_INCLUDE_DIR"
            "-UPython3_LIBRARY"
            "-UPython3_LIBRARY_RELEASE"
            "-UPython3_LIBRARY_DEBUG"
            "-U_Python3_EXECUTABLE"
            "-U_Python3_INCLUDE_DIR"
            "-U_Python3_LIBRARY_RELEASE"
            "-U_Python3_LIBRARY_DEBUG"
            "-UPYBIND11_PYTHON_EXECUTABLE_LAST"
            "-UFIND_PACKAGE_MESSAGE_DETAILS_Python3"
        )
    fi

    echo "[CONFIGURE] openvino"
    cmake -S "${OPENVINO_SRC}" -B "${OPENVINO_BUILD}" -G Ninja "${cmake_args[@]}"
    OPENVINO_ALREADY_CONFIGURED=1
}

build_openvino() {
    echo "[BUILD] openvino"
    configure_openvino
    cmake --build "${OPENVINO_BUILD}" --parallel
}

stage_genai_bin_layout() {
    local genai_bin_dir
    genai_bin_dir="${GENAI_BIN_ROOT}/${BUILD_TYPE}"

    if [[ ! -d "${GENAI_BIN_ROOT}" ]]; then
        log_error "OpenVINO GenAI bin directory not found: ${GENAI_BIN_ROOT}"
        return 1
    fi

    mkdir -p "${genai_bin_dir}"
    find "${genai_bin_dir}" -maxdepth 1 -type f -delete

    while IFS= read -r -d '' file_path; do
        cp -f "${file_path}" "${genai_bin_dir}/"
    done < <(find "${GENAI_BIN_ROOT}" -maxdepth 1 -type f -print0)

    log_info "OpenVINO GenAI top-level bin files staged in: ${genai_bin_dir}"
}

build_genai() {
    if [[ ! -f "${OPENVINO_BUILD}/OpenVINOConfig.cmake" ]]; then
        log_error "CMake configure failed for openvino.genai."
        echo "        Make sure openvino build directory exists and is valid." >&2
        return 1
    fi

    echo "[BUILD] openvino.genai"
    local cmake_args openvino_build_fwd
    openvino_build_fwd="${OPENVINO_BUILD//\\//}"
    cmake_args=("-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" "-DOpenVINO_DIR=${openvino_build_fwd}")
    if [[ "${BUILD_WHEEL}" == "1" ]]; then
        ensure_wheel_python
        cmake_args+=(
            "-DENABLE_PYTHON=ON"
            "-UPython3_EXECUTABLE"
            "-UPython3_ROOT_DIR"
            "-UPython3_INCLUDE_DIR"
            "-UPython3_LIBRARY"
            "-UPython3_LIBRARY_RELEASE"
            "-UPython3_LIBRARY_DEBUG"
            "-U_Python3_EXECUTABLE"
            "-U_Python3_INCLUDE_DIR"
            "-U_Python3_LIBRARY_RELEASE"
            "-U_Python3_LIBRARY_DEBUG"
            "-UPYBIND11_PYTHON_EXECUTABLE_LAST"
            "-UFIND_PACKAGE_MESSAGE_DETAILS_Python3"
            "-DPython3_EXECUTABLE=${WHEEL_PYTHON}"
        )
    else
        cmake_args+=(
            "-DENABLE_PYTHON=OFF"
            "-UPython3_EXECUTABLE"
            "-UPython3_ROOT_DIR"
            "-UPython3_INCLUDE_DIR"
            "-UPython3_LIBRARY"
            "-UPython3_LIBRARY_RELEASE"
            "-UPython3_LIBRARY_DEBUG"
            "-U_Python3_EXECUTABLE"
            "-U_Python3_INCLUDE_DIR"
            "-U_Python3_LIBRARY_RELEASE"
            "-U_Python3_LIBRARY_DEBUG"
            "-UPYBIND11_PYTHON_EXECUTABLE_LAST"
            "-UFIND_PACKAGE_MESSAGE_DETAILS_Python3"
        )
    fi

    cmake -S "${GENAI_SRC}" -B "${GENAI_BUILD}" -G Ninja "${cmake_args[@]}"
    cmake --build "${GENAI_BUILD}" --parallel
    stage_genai_bin_layout
}

build_wheels() {
    ensure_wheel_python
    configure_openvino
    prepare_wheel_output
    local openvino_build_fwd
    openvino_build_fwd="${OPENVINO_BUILD//\\//}"

    echo "[BUILD] openvino wheel"
    cmake --build "${OPENVINO_BUILD}" --config "${BUILD_TYPE}" --target ie_wheel --parallel
    copy_latest_openvino_wheel

    echo "[SETUP] Installing local openvino wheel into the wheel build venv"
    "${WHEEL_PYTHON}" -m pip install --force-reinstall --no-deps "${OPENVINO_WHEEL_PATH}"

    echo "[BUILD] openvino_tokenizers wheel"
    "${WHEEL_PYTHON}" -m pip wheel "${GENAI_SRC}/thirdparty/openvino_tokenizers" \
        --wheel-dir "${WHEEL_OUTPUT_DIR}" \
        --no-deps \
        --no-build-isolation \
        --config-settings="override=cmake.build_path=${GENAI_BUILD}/openvino_tokenizers_wheel_build_${WHEEL_TAG}" \
        --config-settings="override=cmake.options.OpenVINO_DIR=${openvino_build_fwd}" \
        -v

    echo "[BUILD] openvino.genai wheel"
    "${WHEEL_PYTHON}" -m pip wheel "${GENAI_SRC}" \
        --wheel-dir "${WHEEL_OUTPUT_DIR}" \
        --find-links "${WHEEL_OUTPUT_DIR}" \
        --no-deps \
        --no-build-isolation \
        --config-settings="override=cmake.options.OpenVINO_DIR=${openvino_build_fwd}" \
        -v

    echo "[DOWNLOAD] wheel runtime dependencies"
    "${WHEEL_PYTHON}" -m pip download --dest "${WHEEL_OUTPUT_DIR}" --only-binary=:all: 'numpy<2.5.0,>=1.16.6' 'openvino-telemetry>=2023.2.1'

    if [[ ! -f "${WHEEL_SCRIPT}" ]]; then
        log_error "wheel.py not found: ${WHEEL_SCRIPT}"
        return 1
    fi
    cp -f "${WHEEL_SCRIPT}" "${WHEEL_OUTPUT_DIR}/wheel.py"

    local genai_wheel
    genai_wheel="$(find "${WHEEL_OUTPUT_DIR}" -maxdepth 1 -type f -name 'openvino_genai-*.whl' | head -n 1 || true)"
    echo "[OK] Wheel output ready: ${WHEEL_OUTPUT_DIR}"
    if [[ -n "${genai_wheel}" ]]; then
        echo "[INFO] Offline install example:"
        echo "       python -m pip install --no-index --find-links \"${WHEEL_OUTPUT_DIR}\" \"${genai_wheel}\""
        echo "[INFO] Smoke test example:"
        echo "       python \"${WHEEL_OUTPUT_DIR}/wheel.py\" --help"
        echo "       python \"${WHEEL_OUTPUT_DIR}/wheel.py\" --model \"path/to/cached_model.xml\" --device GPU --max-new-tokens 24"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --wheel)
                BUILD_WHEEL=1
                ;;
            --python=*)
                WHEEL_PYTHON_REQUEST="${1#--python=}"
                if [[ -z "${WHEEL_PYTHON_REQUEST}" ]]; then
                    log_error "Missing value for --python."
                    usage
                    exit 1
                fi
                ;;
            --python)
                log_error "Use --python=<version>."
                usage
                exit 1
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log_error "Invalid argument: $1"
                usage
                exit 1
                ;;
        esac
        shift
    done

    if [[ -n "${WHEEL_PYTHON_REQUEST}" && "${BUILD_WHEEL}" != "1" ]]; then
        log_error "--python can only be used together with --wheel."
        usage
        exit 1
    fi
}

main() {
    parse_args "$@"
    ensure_build_tools

    if [[ "${BUILD_OPENVINO}" == "1" ]]; then
        build_openvino
    fi

    if [[ "${BUILD_GENAI}" == "1" ]]; then
        build_genai
    fi

    if [[ "${BUILD_WHEEL}" == "1" ]]; then
        build_wheels
    fi

    echo "[OK] Build finished."
}

main "$@"