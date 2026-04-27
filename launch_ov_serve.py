#!/usr/bin/env python3
"""Launch ov_serve with model-specific defaults from config.json.

Reads config.json (if present) to set model-specific defaults.
CLI arguments override config.json, which overrides hard-coded defaults.

Priority: CLI args > config.json > hard-coded defaults

Examples:
    python launch_ov_serve.py
    python launch_ov_serve.py --no-vl
    python launch_ov_serve.py --model /path/to/Qwen3.5-35B-A3B
    python launch_ov_serve.py --rep-penalty 1.2   # override config.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


IS_WINDOWS = sys.platform == "win32"
EXE_NAME = "ov_serve.exe" if IS_WINDOWS else "ov_serve"
TOKENIZER_LIB_NAME = "openvino_tokenizers.dll" if IS_WINDOWS else "libopenvino_tokenizers.so"
PATH_VAR = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
PATH_SEP = ";" if IS_WINDOWS else ":"
PYTHONPATH_VAR = "PYTHONPATH"


def _print_banner(config: dict[str, object], runtime_dirs: list[Path], log_file: Path | None) -> str:
    lines = []
    lines.append("═══════════════════════════════════════════════════")
    lines.append("  ov_serve — OpenVINO Inference Server")
    lines.append("═══════════════════════════════════════════════════")
    lines.append("")
    lines.append(f"  Launch mode:    {config['launch_mode']}")
    lines.append(f"  Config:         {config.get('config_source', 'defaults')}")
    lines.append(f"  Executable:     {config['exe']}")
    lines.append(f"  Model:          {config['model']}")
    lines.append(f"  Model Name:     {config['model_name']}")
    lines.append(f"  Device:         {config['device']}")
    lines.append(f"  Port:           {config['port']}")
    lines.append(f"  Workers:        {config['workers']}")
    lines.append(f"  Vision (VL):    {config['vl']}")
    lines.append(f"  Thinking:       {config['thinking']}")
    lines.append(f"  Temperature:    {config['temperature']}")
    lines.append(f"  Top P:          {config['top_p']}")
    lines.append(f"  Top K:          {config['top_k']}")
    lines.append(f"  Rep.Penalty:    {config['rep_penalty']}")
    lines.append(f"  Pres.Penalty:   {config['pres_penalty']}")
    lines.append(f"  Freq.Penalty:   {config['freq_penalty']}")
    lines.append(f"  Max Tokens:     {config['max_tokens']}")
    lines.append(f"  Warmup Tokens:  {config['warmup_tokens']}")
    lines.append(f"  Logging:        {config['logging']}")
    lines.append(f"  Quant:          {config['quant_mode']} / group_size={config['quant_group_size']} / backup={config['quant_backup_mode']}")
    lines.append(f"  MoE Prefill:    hybrid={config.get('hybrid_prefill', '1')} gpu_mask={config.get('gpu_mask_prefill', '1')}")
    lines.append("")
    lines.append(f"  {PATH_VAR}:")
    for path in runtime_dirs:
        status = "OK" if path.exists() else "MISSING"
        lines.append(f"    [{status}] {path}")
    lines.append("")
    if log_file is not None:
        lines.append(f"  Log file:       {log_file}")
        lines.append("")
    lines.append("  Connect:")
    lines.append(f"    Local Base URL: http://127.0.0.1:{config['port']}/v1")
    lines.append(f"    LAN Base URL:   http://<server-ip>:{config['port']}/v1")
    lines.append("    API Key:        any non-empty string")
    lines.append("    Model:          default")
    lines.append("")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    return text


def _prepend_env_paths(env: dict[str, str], env_var: str, paths: list[Path]) -> list[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return []
    current = env.get(env_var, "")
    env[env_var] = PATH_SEP.join([str(path) for path in existing] + ([current] if current else []))
    return existing


def _configure_tokenizer_python(env: dict[str, str], script_dir: Path, workspace_root: Path) -> list[Path]:
    python_paths: list[Path] = []
    tokenizer_lib_candidates = [
        script_dir / TOKENIZER_LIB_NAME,
        workspace_root / "openvino.genai" / "build" / "bin" / "Release" / TOKENIZER_LIB_NAME,
        workspace_root / "openvino.genai" / "build" / "bin" / TOKENIZER_LIB_NAME,
    ]
    tokenizer_python_candidates = [
        script_dir if (script_dir / "openvino_tokenizers").is_dir() else None,
        workspace_root / "openvino.genai" / "thirdparty" / "openvino_tokenizers" / "python",
    ]

    tokenizer_lib = next((path for path in tokenizer_lib_candidates if path.exists()), None)
    if tokenizer_lib is not None:
        env["OV_TOKENIZER_PREBUILD_EXTENSION_PATH"] = str(tokenizer_lib)

    for path in tokenizer_python_candidates:
        if path is not None and path.exists():
            python_paths.append(path)
    return _prepend_env_paths(env, PYTHONPATH_VAR, python_paths)


def _candidate_model_paths(script_dir: Path, workspace_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_model = os.environ.get("OV_SERVE_MODEL")
    if env_model:
        candidates.append(Path(env_model))

    bundled_models_root = script_dir / "models"
    if bundled_models_root.is_dir():
        preferred = [
            bundled_models_root / "Qwen3.5-9B",
            bundled_models_root / "Qwen3.5-4B",
            bundled_models_root / "Qwen3.5-35B-A3B",
        ]
        for candidate in preferred:
            if candidate not in candidates:
                candidates.append(candidate)
        for child in sorted(bundled_models_root.iterdir()):
            if child.is_dir() and child not in candidates:
                candidates.append(child)

    candidates.extend(
        [
            workspace_root.parent / "models" / "Huggingface" / "Qwen3.5-4B",
            workspace_root.parent / "models" / "Huggingface" / "Qwen3.5-9B",
            Path.home() / "models" / "Huggingface" / "Qwen3.5-4B",
            Path.home() / "models" / "Huggingface" / "Qwen3.5-9B",
        ]
    )
    if IS_WINDOWS:
        candidates.append(Path(r"C:\data\models\Huggingface\Qwen3.5-4B"))
        candidates.append(Path(r"C:\data\models\Huggingface\Qwen3.5-9B"))
    return candidates


def _default_model(script_dir: Path, workspace_root: Path) -> Path | None:
    for candidate in _candidate_model_paths(script_dir, workspace_root):
        if candidate.exists():
            return candidate
    return None


def _resolve_build_tree(script_dir: Path) -> tuple[Path, list[Path]]:
    workspace_root = script_dir.parent
    ov_root = workspace_root / "openvino"
    genai_root = workspace_root / "openvino.genai"

    exe_candidates = [
        genai_root / "build" / "bin" / "Release" / EXE_NAME,
        genai_root / "build" / "bin" / EXE_NAME,
    ]
    exe = next((candidate for candidate in exe_candidates if candidate.exists()), None)
    if exe is None:
        joined = "\n".join(f"  - {candidate}" for candidate in exe_candidates)
        raise FileNotFoundError(f"{EXE_NAME} not found. Build first. Checked:\n{joined}")

    runtime_dirs = [
        ov_root / "bin" / "intel64" / "Release",
        ov_root / "temp" / ("Windows_AMD64" if IS_WINDOWS else "Linux_x86_64") / "tbb" / ("bin" if IS_WINDOWS else "lib"),
        ov_root / "build" / "bin" / "Release",
        genai_root / "build" / "openvino_genai",
        genai_root / "build" / "bin" / "Release",
        genai_root / "build" / "bin",
    ]
    return exe, runtime_dirs


def _resolve_launch_layout(script_dir: Path) -> tuple[str, Path, list[Path], Path]:
    standalone_exe = script_dir / EXE_NAME
    workspace_root = script_dir.parent
    if standalone_exe.exists():
        return "standalone", standalone_exe, [script_dir], workspace_root
    exe, runtime_dirs = _resolve_build_tree(script_dir)
    return "build-tree", exe, runtime_dirs, workspace_root


# ── Config file support ──────────────────────────────────────────────

# Hard-coded defaults (used when neither config.json nor CLI provides a value)
_DEFAULTS = {
    "port": 8080,
    "warmup_tokens": 512,
    "device": "GPU",
    "workers": 1,
    "temperature": 0.1,
    "top_p": 1.0,
    "top_k": 20,
    "rep_penalty": 1.0,
    "pres_penalty": 0.0,
    "freq_penalty": 0.0,
    "min_temp": 0.0,
    "max_tokens": 2048,
    "group_size": 128,
    "vl": True,
    "thinking": False,
    "log": False,
    "quant_mode": "int4_asym",
    "backup_mode": "int8_asym",
}


def _load_config(script_dir: Path) -> dict:
    """Load config.json from script_dir if it exists, return empty dict otherwise."""
    config_path = script_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"  Loaded config: {config_path}")
        return cfg
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to load {config_path}: {e}")
        return {}


def _merge_defaults(file_cfg: dict) -> dict:
    """Merge file config over hard-coded defaults. Returns merged defaults dict."""
    merged = dict(_DEFAULTS)
    # Map config.json keys to internal keys (config.json uses the same names)
    for key in _DEFAULTS:
        if key in file_cfg:
            merged[key] = file_cfg[key]
    return merged


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    d = defaults
    parser = argparse.ArgumentParser(description="Launch ov_serve with model-specific defaults from config.json.")
    parser.add_argument("--model", type=Path, default=None, help="Path to HF model directory.")
    parser.add_argument("--no-vl", action="store_true", default=not d["vl"], help="Disable vision-language mode.")
    parser.add_argument("--port", type=int, default=d["port"], help=f"HTTP port (default: {d['port']}).")
    parser.add_argument("--warmup-tokens", type=int, default=d["warmup_tokens"], help=f"Warmup sequence length (default: {d['warmup_tokens']}).")
    parser.add_argument("--thinking", action="store_true", default=d["thinking"], help="Enable thinking mode.")
    parser.add_argument("--device", default=d["device"], help=f"Target device (default: {d['device']}).")
    parser.add_argument("--workers", type=int, default=d["workers"], help=f"Worker count (default: {d['workers']}).")
    parser.add_argument("--temperature", type=float, default=d["temperature"], help=f"Default temperature (default: {d['temperature']}).")
    parser.add_argument("--top-p", type=float, default=d["top_p"], help=f"Default top-p (default: {d['top_p']}).")
    parser.add_argument("--top-k", type=int, default=d["top_k"], help=f"Default top-k (default: {d['top_k']}).")
    parser.add_argument("--rep-penalty", type=float, default=d["rep_penalty"], help=f"Repetition penalty (default: {d['rep_penalty']}).")
    parser.add_argument("--pres-penalty", type=float, default=d["pres_penalty"], help=f"Presence penalty (default: {d['pres_penalty']}).")
    parser.add_argument("--freq-penalty", type=float, default=d["freq_penalty"], help=f"Frequency penalty (default: {d['freq_penalty']}).")
    parser.add_argument("--min-temp", type=float, default=d["min_temp"], help=f"Minimum sampling temperature (default: {d['min_temp']}).")
    parser.add_argument("--max-tokens", type=int, default=d["max_tokens"], help=f"Maximum generated tokens (default: {d['max_tokens']}).")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for /v1/models (default: directory name or config).")
    parser.add_argument("--group-size", type=int, default=d["group_size"], help=f"Quantization group size (default: {d['group_size']}).")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--log", action="store_true", dest="log",
                           help="Enable stderr logging to ov_serve.log.")
    log_group.add_argument("--no-log", action="store_false", dest="log",
                           help="Disable stderr logging.")
    parser.set_defaults(log=d["log"])
    return parser


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent

    # Load config.json → merge with hard-coded defaults → parse CLI (CLI wins)
    file_cfg = _load_config(script_dir)
    defaults = _merge_defaults(file_cfg)
    args = build_parser(defaults).parse_args(argv)

    launch_mode, exe, runtime_dirs, workspace_root = _resolve_launch_layout(script_dir)

    model = args.model or _default_model(script_dir, workspace_root)
    if model is None:
        raise SystemExit("No default model path was found. Pass --model explicitly.")
    if not model.exists():
        raise SystemExit(f"Model directory not found: {model}")
    model_name = args.model_name or file_cfg.get("model_name") or model.name

    quant_mode = file_cfg.get("quant_mode", defaults["quant_mode"])
    backup_mode = file_cfg.get("backup_mode", defaults["backup_mode"])
    group_size_str = str(args.group_size)

    env = os.environ.copy()
    env["OV_GENAI_USE_MODELING_API"] = "1"
    env.setdefault("OV_GENAI_INFLIGHT_QUANT_MODE", quant_mode)
    env.setdefault("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", group_size_str)
    env.setdefault("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", backup_mode)
    env.setdefault("MOE_USE_HYBRID_PREFILL", "1")
    env.setdefault("MOE_USE_GPU_MASK_PREFILL", "1")
    resolved_runtime_dirs = _prepend_env_paths(env, PATH_VAR, runtime_dirs)
    _configure_tokenizer_python(env, script_dir, workspace_root)

    cmd = [
        str(exe),
        "--model",
        str(model),
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--rep-penalty",
        str(args.rep_penalty),
        "--pres-penalty",
        str(args.pres_penalty),
        "--freq-penalty",
        str(args.freq_penalty),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--model-name",
        model_name,
    ]
    if not args.no_vl:
        cmd.append("--vl")
    if not args.thinking:
        cmd.append("--no-thinking")
    if args.min_temp > 0:
        cmd.extend(["--min-temp", str(args.min_temp)])
    if not args.log:
        cmd.append("--no-log")

    log_file = script_dir / "ov_serve.log" if args.log else None
    config_source = "config.json" if file_cfg else "built-in defaults"
    banner_text = _print_banner(
        {
            "launch_mode": launch_mode,
            "config_source": config_source,
            "exe": exe,
            "model": model,
            "model_name": model_name,
            "device": args.device,
            "port": args.port,
            "workers": args.workers,
            "vl": not args.no_vl,
            "thinking": args.thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "rep_penalty": args.rep_penalty,
            "pres_penalty": args.pres_penalty,
            "freq_penalty": args.freq_penalty,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "logging": args.log,
            "quant_mode": env.get("OV_GENAI_INFLIGHT_QUANT_MODE", quant_mode),
            "quant_group_size": env.get("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", group_size_str),
            "quant_backup_mode": env.get("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", backup_mode),
            "hybrid_prefill": env.get("MOE_USE_HYBRID_PREFILL", "1"),
            "gpu_mask_prefill": env.get("MOE_USE_GPU_MASK_PREFILL", "1"),
        },
        resolved_runtime_dirs,
        log_file,
    )

    if log_file is None:
        completed = subprocess.run(cmd, env=env, check=False)
    else:
        with log_file.open("w", encoding="utf-8") as log_handle:
            # Write the banner to the log so the config is recorded.
            log_handle.write(banner_text)
            log_handle.flush()
            # Redirect both stdout and stderr to the log file so that
            # all output ([ModelLoader], [ov_serve], [GPU] messages)
            # is captured in a single file.
            completed = subprocess.run(
                cmd, env=env, check=False,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
