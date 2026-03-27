"""download_models.py – Download all models required by auto_tests.py.

Usage:
    python download_models.py [--models-root D:\\data\\models] [--hf-token <token>]
                              [--dry-run] [--skip-existing] [--list]
                              [--only <name1> [<name2> ...]]

Options:
    --models-root   Root directory for model storage (default: D:\\data\\models)
    --hf-token      HuggingFace access token (or set HF_TOKEN env var)
    --dry-run       Print what would be downloaded without actually downloading
    --skip-existing Skip models whose target directory / file already exists (default: True)
    --no-skip       Force re-download even if the target already exists
    --list          List all models and exit
    --only          Download only the specified model keys (space-separated)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Huggingface full-repository downloads: local_dir_name -> HF repo id
# Update the repo IDs if the public names differ from the local directory names.
HF_MODELS: Dict[str, str] = {
    # --- Qwen3 text models ---
    "Qwen3-0.6B":                               "Qwen/Qwen3-0.6B",
    "Qwen3-4B":                                 "Qwen/Qwen3-4B",
    "Qwen3-8B":                                 "Qwen/Qwen3-8B",
    "Qwen3-30B-A3B-Instruct-2507":              "Qwen/Qwen3-30B-A3B-Instruct-2507",
    # --- Qwen3 vision-language models ---
    "Qwen3-VL-2B-Instruct":                     "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct":                     "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct":                     "Qwen/Qwen3-VL-8B-Instruct",
    # --- Qwen3 speech models ---
    "Qwen3-TTS-12Hz-1.7B-Base":                 "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-ASR-0.6B":                           "Qwen/Qwen3-ASR-0.6B",
    # --- Qwen3.5 models ---
    "Qwen3.5-0.8B":                             "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B":                               "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B":                               "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B":                               "Qwen/Qwen3.5-9B",
    "Qwen3.5-27B":                              "Qwen/Qwen3.5-27B",
    "Qwen3.5-35B-A3B-Base":                     "Qwen/Qwen3.5-35B-A3B-Base",
    "Qwen3.5-35B-A3B":                          "Qwen/Qwen3.5-35B-A3B",
    # --- Other text models ---
    "SmolLM3-3B":                               "HuggingFaceTB/SmolLM3-3B",
    # --- OCR models ---
    "DeepSeek-OCR-2":                           "deepseek-ai/DeepSeek-OCR-2",
    "GLM-OCR":                                  "zai-org/GLM-OCR",
    # --- Image / video generation ---
    "Z-Image-Turbo":                            "Tongyi-MAI/Z-Image-Turbo",
    "Wan2.1-T2V-1.3B-Diffusers":               "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    # --- Speculative decoding draft model ---
    "Qwen3-4B-DFlash-b16":                     "z-lab/Qwen3-4B-DFlash-b16",
    # --- MoE model ---
    "Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2": "DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2",
    # --- Internal / private models (update repo IDs before use) ---
    "Youtu-LLM-2B":                             "tencent/Youtu-LLM-2B",
}

# GGUF single-file downloads: local_filename -> (HF repo id, filename in repo)
# Files are saved to <models_root>/gguf/<local_filename>.
# Note: BF16/Q4_0 variants are not publicly available on HF; the closest public
# quantization is used and saved under the filename expected by auto_tests.py.
GGUF_MODELS: Dict[str, Tuple[str, str]] = {
    "Qwen3-0.6B-BF16.gguf":                    ("Qwen/Qwen3-0.6B-GGUF",    "Qwen3-0.6B-Q8_0.gguf"),
    "Qwen3-4B-BF16.gguf":                      ("Qwen/Qwen3-4B-GGUF",      "Qwen3-4B-Q8_0.gguf"),
    "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf":   ("Qwen/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf"),
}

PLACEHOLDER_PREFIX = "PLACEHOLDER"


def _check_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       Run: pip install huggingface_hub")
        sys.exit(1)


def _login(token: Optional[str]) -> None:
    from huggingface_hub import login
    if token:
        login(token=token, add_to_git_credential=False)


def _placeholder_repos(models: List[str]) -> List[str]:
    return [k for k in models if HF_MODELS.get(k, "").startswith(PLACEHOLDER_PREFIX)]


def download_hf_model(
    key: str,
    repo_id: str,
    local_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> bool:
    """Download a full HuggingFace repo into local_dir. Returns True on success."""
    from huggingface_hub import snapshot_download

    if skip_existing and local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [SKIP]  {key}  (already exists at {local_dir})")
        return True

    if dry_run:
        print(f"  [DRY]   {key}  ->  {repo_id}  ->  {local_dir}")
        return True

    print(f"  [DOWN]  {key}  from  {repo_id}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        print(f"          -> saved to {local_dir}")
        return True
    except Exception as exc:
        print(f"  [FAIL]  {key}: {exc}")
        return False


def download_gguf_file(
    filename: str,
    repo_id: str,
    repo_filename: str,
    local_path: Path,
    skip_existing: bool,
    dry_run: bool,
) -> bool:
    """Download a single GGUF file. Returns True on success."""
    from huggingface_hub import hf_hub_download

    if skip_existing and local_path.exists():
        print(f"  [SKIP]  {filename}  (already exists at {local_path})")
        return True

    if dry_run:
        print(f"  [DRY]   {filename}  ->  {repo_id}/{repo_filename}  ->  {local_path}")
        return True

    print(f"  [DOWN]  {filename}  from  {repo_id}/{repo_filename}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=repo_filename,
            local_dir=str(local_path.parent),
        )
        # Rename to the expected local filename if it differs from the repo filename
        downloaded_path = Path(downloaded)
        if downloaded_path.name != local_path.name:
            downloaded_path.rename(local_path)
            print(f"          -> renamed {downloaded_path.name} -> {local_path.name}")
        print(f"          -> saved to {local_path}")
        return True
    except Exception as exc:
        print(f"  [FAIL]  {filename}: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all models required by auto_tests.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-root",
        default=r"D:\data\models",
        help="Root directory for models (default: D:\\data\\models)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip models that already exist locally (default)",
    )
    skip_group.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download even if the model already exists locally",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models and their HF repo IDs, then exit",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="Download only the specified model keys",
    )
    return parser.parse_args()


def list_models() -> None:
    print("\nHuggingFace full-repository models:")
    print(f"  {'Local directory':<52}  HF repo ID")
    print(f"  {'-'*52}  {'-'*40}")
    for key, repo_id in HF_MODELS.items():
        flag = " [PLACEHOLDER]" if repo_id.startswith(PLACEHOLDER_PREFIX) else ""
        print(f"  {key:<52}  {repo_id}{flag}")

    print("\nGGUF single-file models:")
    print(f"  {'Local filename':<52}  HF repo / file")
    print(f"  {'-'*52}  {'-'*40}")
    for filename, (repo_id, repo_file) in GGUF_MODELS.items():
        print(f"  {filename:<52}  {repo_id}/{repo_file}")


def main() -> int:
    args = parse_args()

    if args.list:
        list_models()
        return 0

    skip_existing = not args.no_skip

    _check_huggingface_hub()

    models_root = Path(args.models_root)
    hf_dir = models_root / "Huggingface"
    gguf_dir = models_root / "gguf"

    # Determine which HF models to download
    hf_keys = list(HF_MODELS.keys())
    gguf_keys = list(GGUF_MODELS.keys())

    if args.only:
        requested = set(args.only)
        hf_keys = [k for k in hf_keys if k in requested]
        gguf_keys = [k for k in gguf_keys if k in requested]
        unknown = requested - set(hf_keys) - set(gguf_keys) - set(HF_MODELS) - set(GGUF_MODELS)
        if unknown:
            print(f"WARNING: Unknown model key(s): {', '.join(sorted(unknown))}")

    # Warn about placeholder repos
    placeholders = _placeholder_repos(hf_keys)
    if placeholders:
        print("\nWARNING: The following models have PLACEHOLDER repo IDs.")
        print("         Update HF_MODELS in this script before running:")
        for k in placeholders:
            print(f"  {k}")
        print()

    if args.hf_token:
        print("Logging into HuggingFace Hub ...")
        _login(args.hf_token)

    successes: List[str] = []
    failures: List[str] = []
    skipped: List[str] = []

    # --- HuggingFace full-repo models ---
    if hf_keys:
        print(f"\n{'='*60}")
        print(f"Downloading {len(hf_keys)} HuggingFace model(s) into {hf_dir}")
        print(f"{'='*60}")
        hf_dir.mkdir(parents=True, exist_ok=True)

        for key in hf_keys:
            repo_id = HF_MODELS[key]
            if repo_id.startswith(PLACEHOLDER_PREFIX):
                print(f"  [SKIP]  {key}  (placeholder repo ID – update HF_MODELS)")
                skipped.append(key)
                continue
            local_dir = hf_dir / key
            ok = download_hf_model(key, repo_id, local_dir, skip_existing, args.dry_run)
            (successes if ok else failures).append(key)

    # --- GGUF single-file models ---
    if gguf_keys:
        print(f"\n{'='*60}")
        print(f"Downloading {len(gguf_keys)} GGUF file(s) into {gguf_dir}")
        print(f"{'='*60}")
        gguf_dir.mkdir(parents=True, exist_ok=True)

        for filename in gguf_keys:
            repo_id, repo_file = GGUF_MODELS[filename]
            local_path = gguf_dir / filename
            ok = download_gguf_file(
                filename, repo_id, repo_file, local_path, skip_existing, args.dry_run
            )
            (successes if ok else failures).append(filename)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Downloaded : {len(successes)}")
    print(f"  Skipped    : {len(skipped)}")
    print(f"  Failed     : {len(failures)}")
    if failures:
        print("\nFailed models:")
        for name in failures:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
