#!/usr/bin/env python3
"""Launch ov_serve with cross-platform defaults for Qwen3.5.

Examples:
    python launch_ov_serve.py
    python launch_ov_serve.py --no-vl
    python launch_ov_serve.py --model /path/to/Qwen3.5-35B-A3B
"""

from __future__ import annotations

import argparse
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
    lines.append("  ov_serve — Qwen3.5 OpenVINO Inference Server")
    lines.append("═══════════════════════════════════════════════════")
    lines.append("")
    lines.append(f"  Launch mode:    {config['launch_mode']}")
    lines.append(f"  Executable:     {config['exe']}")
    lines.append(f"  Model:          {config['model']}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch ov_serve with Qwen3.5 defaults.")
    parser.add_argument("--model", type=Path, default=None, help="Path to HF model directory.")
    parser.add_argument("--no-vl", action="store_true", help="Disable vision-language mode.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--warmup-tokens", type=int, default=512, help="Warmup sequence length (0 disables warmup).")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode.")
    parser.add_argument("--device", default="GPU", help="Target device.")
    parser.add_argument("--workers", type=int, default=1, help="Worker count.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Default temperature when requests omit it.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Default top-p when requests omit it.")
    parser.add_argument("--top-k", type=int, default=20, help="Default top-k when requests omit it.")
    parser.add_argument("--rep-penalty", type=float, default=1.0, help="Repetition penalty (1.0 = off).")
    parser.add_argument("--pres-penalty", type=float, default=0.0, help="Presence penalty.")
    parser.add_argument("--freq-penalty", type=float, default=0.0, help="Frequency penalty.")
    parser.add_argument("--min-temp", type=float, default=0.0, help="Minimum sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum generated tokens.")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size (e.g. 32, 128).")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--log", action="store_true", dest="log",
                           help="Enable stderr logging to ov_serve.log.")
    log_group.add_argument("--no-log", action="store_false", dest="log",
                           help="Disable stderr logging (default).")
    parser.set_defaults(log=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    args = build_parser().parse_args(argv)
    launch_mode, exe, runtime_dirs, workspace_root = _resolve_launch_layout(script_dir)

    model = args.model or _default_model(script_dir, workspace_root)
    if model is None:
        raise SystemExit("No default model path was found. Pass --model explicitly.")
    if not model.exists():
        raise SystemExit(f"Model directory not found: {model}")

    env = os.environ.copy()
    env["OV_GENAI_USE_MODELING_API"] = "1"
    env.setdefault("OV_GENAI_INFLIGHT_QUANT_MODE", "int4_asym")
    env.setdefault("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", str(args.group_size))
    env.setdefault("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", "int4_sym")
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
    banner_text = _print_banner(
        {
            "launch_mode": launch_mode,
            "exe": exe,
            "model": model,
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
            "quant_mode": env.get("OV_GENAI_INFLIGHT_QUANT_MODE", "int4_asym"),
            "quant_group_size": env.get("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", "128"),
            "quant_backup_mode": env.get("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", "int4_sym"),
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
