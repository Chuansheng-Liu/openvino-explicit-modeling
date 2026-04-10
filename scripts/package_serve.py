"""Package ov_serve, convert_ir, and modeling_qwen3_5_cli into a standalone directory.

Collects only the executables and DLLs needed for serving, so the
resulting folder can be copied to any machine and run without a full
build tree.

Usage:
    python scripts\\package_serve.py
    python scripts\\package_serve.py --clean
    python scripts\\package_serve.py --output D:\\deploy\\ov_serve_bundle
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Executables to include
# ---------------------------------------------------------------------------
SERVE_EXECUTABLES = {
    "ov_serve.exe",
    "convert_ir.exe",
    "modeling_qwen3_5_cli.exe",
}

# ---------------------------------------------------------------------------
# Source specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileSource:
    """Describes a set of files to collect from the build tree."""
    name: str
    relative_path_template: str
    source_kind: str          # "directory" or "file"
    suffixes: tuple[str, ...] = (".dll",)
    name_filter: frozenset[str] | None = None  # None = take all matching suffix

    def resolve(self, workspace_root: Path, config: str) -> Path:
        return workspace_root / Path(self.relative_path_template.format(config=config))


# Order matters only for logging; all files end up in one flat directory.
SOURCES = (
    # --- Executables (cherry-picked) ---
    FileSource(
        name="Serve executables",
        relative_path_template="openvino.genai/build/bin/{config}",
        source_kind="directory",
        suffixes=(".exe",),
        name_filter=frozenset(SERVE_EXECUTABLES),
    ),
    # --- OpenVINO runtime DLLs ---
    FileSource(
        name="OpenVINO runtime DLLs",
        relative_path_template="openvino/bin/intel64/{config}",
        source_kind="directory",
        suffixes=(".dll",),
    ),
    # --- TBB ---
    FileSource(
        name="TBB runtime DLL",
        relative_path_template="openvino/temp/Windows_AMD64/tbb/bin/tbb12.dll",
        source_kind="file",
        suffixes=(".dll",),
    ),
    # --- OpenVINO GenAI DLLs ---
    FileSource(
        name="OpenVINO GenAI DLLs",
        relative_path_template="openvino.genai/build/openvino_genai",
        source_kind="directory",
        suffixes=(".dll",),
    ),
    # --- OpenCV + tokenizers DLLs from build/bin ---
    FileSource(
        name="OpenCV / tokenizers DLLs",
        relative_path_template="openvino.genai/build/bin",
        source_kind="directory",
        suffixes=(".dll",),
    ),
    # --- Launch script ---
    FileSource(
        name="Launch script",
        relative_path_template="openvino-explicit-modeling/launch_ov_serve.ps1",
        source_kind="file",
        suffixes=(".ps1",),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def fmt(size: int) -> str:
    s = float(size)
    for u in ("B", "KB", "MB", "GB"):
        if s < 1024.0 or u == "GB":
            return f"{s:.2f} {u}" if u != "B" else f"{int(s)} {u}"
        s /= 1024.0
    return f"{size} B"


def collect(src: FileSource, ws: Path, cfg: str) -> tuple[list[Path], list[str]]:
    path = src.resolve(ws, cfg)
    ok_sfx = {s.lower() for s in src.suffixes}

    if src.source_kind == "file":
        if not path.exists():
            return [], [f"{src.name}: not found: {path}"]
        if not path.is_file():
            return [], [f"{src.name}: not a file: {path}"]
        return [path], []

    if not path.is_dir():
        return [], [f"{src.name}: directory not found: {path}"]

    files = sorted(
        f for f in path.iterdir()
        if f.is_file()
        and f.suffix.lower() in ok_sfx
        and (src.name_filter is None or f.name in src.name_filter)
    )
    if not files:
        return [], [f"{src.name}: no matching files in {path}"]
    return files, []


def copy_file(src: Path, dst_dir: Path) -> tuple[str, int]:
    dst = dst_dir / src.name
    sz = src.stat().st_size

    if dst.exists() and dst.is_file():
        if filecmp.cmp(src, dst, shallow=False):
            log("SKIP", f"{src.name} (identical, {fmt(sz)})")
            return "skipped", 0
        log("COPY", f"{src.name} -> overwrite ({fmt(sz)})")
        shutil.copy2(src, dst)
        return "overwritten", sz

    log("COPY", f"{src.name} ({fmt(sz)})")
    shutil.copy2(src, dst)
    return "copied", sz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Package ov_serve and dependencies into a standalone directory.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts\\package_serve.py\n"
            "  python scripts\\package_serve.py --clean\n"
            "  python scripts\\package_serve.py --output D:\\deploy\\serve_bundle\n"
        ),
    )
    p.add_argument("--clean", action="store_true",
                   help="Remove existing files in destination before copying.")
    p.add_argument("--output", type=str, default=None, metavar="DIR",
                   help="Override output root (default: <workspace>/package_serve).")
    p.add_argument("--build-type", choices=("Release", "RelWithDebInfo"),
                   default="Release", help="Build configuration (default: Release).")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    ws = repo_root.parent  # workspace root (explicit_modeling/)
    cfg = args.build_type

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ws / out
        pkg_dir = out.resolve() / cfg
    else:
        pkg_dir = ws / "package_serve" / cfg

    pkg_dir.mkdir(parents=True, exist_ok=True)

    log("INFO", f"Workspace : {ws}")
    log("INFO", f"Config    : {cfg}")
    log("INFO", f"Output    : {pkg_dir}")
    log("INFO", f"Executables: {', '.join(sorted(SERVE_EXECUTABLES))}")

    stats = dict(matched=0, copied=0, overwritten=0, skipped=0, errors=0,
                 matched_bytes=0, written_bytes=0, cleaned=0, cleaned_bytes=0)

    # Clean
    if args.clean:
        for f in pkg_dir.iterdir():
            if f.is_file() and f.suffix.lower() in {".dll", ".exe", ".ps1"}:
                sz = f.stat().st_size
                log("CLEAN", f"{f.name} ({fmt(sz)})")
                f.unlink()
                stats["cleaned"] += 1
                stats["cleaned_bytes"] += sz
        log("INFO", f"Cleaned {stats['cleaned']} file(s), {fmt(stats['cleaned_bytes'])}")

    # Collect and copy
    for src in SOURCES:
        files, issues = collect(src, ws, cfg)
        if issues:
            for iss in issues:
                log("ERROR", iss)
            stats["errors"] += len(issues)
            continue

        log("INFO", f"{src.name}: {len(files)} file(s)")
        for f in files:
            stats["matched"] += 1
            stats["matched_bytes"] += f.stat().st_size
            try:
                action, written = copy_file(f, pkg_dir)
            except Exception as e:
                log("ERROR", f"Failed to copy {f}: {e}")
                stats["errors"] += 1
                continue
            if action == "copied":
                stats["copied"] += 1
                stats["written_bytes"] += written
            elif action == "overwritten":
                stats["overwritten"] += 1
                stats["written_bytes"] += written
            else:
                stats["skipped"] += 1

    # Summary
    final_files = [f for f in pkg_dir.iterdir() if f.is_file()]
    final_bytes = sum(f.stat().st_size for f in final_files)
    exes = [f.name for f in final_files if f.suffix.lower() == ".exe"]
    dlls = [f.name for f in final_files if f.suffix.lower() == ".dll"]

    log("SUMMARY", "=" * 60)
    log("SUMMARY", f"Destination  : {pkg_dir}")
    log("SUMMARY", f"Matched      : {stats['matched']} files ({fmt(stats['matched_bytes'])})")
    log("SUMMARY", f"Copied       : {stats['copied']}, overwritten: {stats['overwritten']}, skipped: {stats['skipped']}")
    log("SUMMARY", f"Errors       : {stats['errors']}")
    log("SUMMARY", f"Written      : {fmt(stats['written_bytes'])}")
    log("SUMMARY", f"Package total: {len(final_files)} files ({fmt(final_bytes)})")
    log("SUMMARY", f"Executables  : {', '.join(sorted(exes))}")
    log("SUMMARY", f"DLLs         : {len(dlls)}")
    log("SUMMARY", "=" * 60)

    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
