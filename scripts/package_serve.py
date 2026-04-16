"""Package ov_serve and optional model files into a standalone directory.

Collects the serving executables, runtime libraries, Python helpers, and
optionally a preconverted model under models/<name> so the resulting
folder can be copied to another machine and run without a full build tree.

Usage:
    python scripts\\package_serve.py
    python scripts\\package_serve.py --clean
    python scripts\\package_serve.py --output D:\\deploy\\ov_serve_bundle
    python scripts\\package_serve.py --model-dir /models/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Executables to include
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"

SERVE_EXECUTABLES = frozenset({
    "ov_serve.exe" if IS_WINDOWS else "ov_serve",
    "convert_ir.exe" if IS_WINDOWS else "convert_ir",
    "modeling_qwen3_5_cli.exe" if IS_WINDOWS else "modeling_qwen3_5_cli",
})

RUNTIME_SUFFIXES = (".dll",) if IS_WINDOWS else (".so",)
TBB_SUFFIXES = (".dll",) if IS_WINDOWS else (".12", ".2")
PACKAGE_SCRIPT_SUFFIXES = (".py",)
PACKAGE_CLEAN_SUFFIXES = {".dll", ".exe", ".ps1", ".py", ".so", ".12", ".2"}
TBB_RELATIVE = "openvino/temp/Windows_AMD64/tbb/bin" if IS_WINDOWS else "openvino/temp/Linux_x86_64/tbb/lib"
TBB_NAMES = frozenset({"tbb12.dll"} if IS_WINDOWS else {"libtbb.so.12", "libtbbmalloc.so.2"})
LAUNCH_SCRIPT_RELATIVE = "openvino-explicit-modeling/launch_ov_serve.py"
MODEL_METADATA_FILES = frozenset({
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
})

# ---------------------------------------------------------------------------
# Source specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileSource:
    """Describes a set of files to collect from the build tree."""
    name: str
    relative_path_template: str
    source_kind: str          # "directory", "file", or "tree"
    suffixes: tuple[str, ...] = (".dll",)
    name_filter: frozenset[str] | None = None  # None = take all matching suffix
    destination_subdir: str | None = None

    def resolve(self, workspace_root: Path, config: str) -> Path:
        return workspace_root / Path(self.relative_path_template.format(config=config))


# Order matters only for logging; all files end up in one flat directory.
BASE_SOURCES = (
    # --- Executables (cherry-picked) ---
    FileSource(
        name="Serve executables",
        relative_path_template="openvino.genai/build/bin/{config}",
        source_kind="directory",
        suffixes=(".exe",) if IS_WINDOWS else ("",),
        name_filter=SERVE_EXECUTABLES,
    ),
    # --- OpenVINO runtime DLLs ---
    FileSource(
        name="OpenVINO runtime libraries",
        relative_path_template="openvino/bin/intel64/{config}",
        source_kind="directory",
        suffixes=RUNTIME_SUFFIXES,
    ),
    # --- TBB ---
    FileSource(
        name="TBB runtime libraries",
        relative_path_template=TBB_RELATIVE,
        source_kind="directory",
        suffixes=TBB_SUFFIXES,
        name_filter=TBB_NAMES,
    ),
    # --- OpenVINO GenAI runtime ---
    FileSource(
        name="OpenVINO GenAI runtime",
        relative_path_template="openvino.genai/build/openvino_genai",
        source_kind="directory",
        suffixes=RUNTIME_SUFFIXES,
    ),
    # --- Tokenizers runtime from build/bin ---
    FileSource(
        name="Tokenizers runtime",
        relative_path_template="openvino.genai/build/bin",
        source_kind="directory",
        suffixes=RUNTIME_SUFFIXES,
        name_filter=frozenset({"libopenvino_tokenizers.so"}) if not IS_WINDOWS else None,
    ),
    # --- Launch script ---
    FileSource(
        name="Launch script",
        relative_path_template=LAUNCH_SCRIPT_RELATIVE,
        source_kind="file",
        suffixes=PACKAGE_SCRIPT_SUFFIXES,
    ),
)

TOKENIZER_FALLBACK_SOURCES = (
    FileSource(
        name="OpenVINO tokenizers Python package",
        relative_path_template="openvino.genai/thirdparty/openvino_tokenizers/python/openvino_tokenizers",
        source_kind="tree",
        suffixes=(".py",),
        destination_subdir="openvino_tokenizers",
    ),
    FileSource(
        name="OpenVINO Python package",
        relative_path_template="openvino/build_python3.12/site-packages/python/openvino",
        source_kind="tree",
        suffixes=(".py", ".pyi", ".so"),
        destination_subdir="openvino",
    ),
)

ALL_SOURCES = BASE_SOURCES + TOKENIZER_FALLBACK_SOURCES


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


def _read_elf_soname(path: Path) -> str | None:
    """Read the SONAME from an ELF shared library, or return None."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"\x7fELF":
                return None
            ei_class = struct.unpack("B", f.read(1))[0]
            is64 = ei_class == 2
            f.seek(0)
            hdr_fmt = "<16sHHIQQQIHHHHHH" if is64 else "<16sHHIIIIIHHHHHH"
            hdr_sz = struct.calcsize(hdr_fmt)
            hdr = struct.unpack(hdr_fmt, f.read(hdr_sz))
            e_phoff = hdr[5] if is64 else hdr[5]
            e_phentsize = hdr[9]
            e_phnum = hdr[10]
            # Find PT_DYNAMIC
            ptr_fmt = "Q" if is64 else "I"
            ptr_sz = struct.calcsize(ptr_fmt)
            dyn_off = dyn_sz = 0
            for i in range(e_phnum):
                f.seek(e_phoff + i * e_phentsize)
                p_type = struct.unpack("<I", f.read(4))[0]
                if p_type == 2:  # PT_DYNAMIC
                    if is64:
                        f.seek(e_phoff + i * e_phentsize + 8)
                        dyn_off = struct.unpack("<Q", f.read(8))[0]
                        dyn_sz = struct.unpack("<Q", f.read(8))[0]
                    else:
                        f.seek(e_phoff + i * e_phentsize + 4)
                        dyn_off = struct.unpack("<I", f.read(4))[0]
                        dyn_sz = struct.unpack("<I", f.read(4))[0]
                    break
            if not dyn_off:
                return None
            # Read dynamic entries to find DT_SONAME and DT_STRTAB
            dyn_entry_fmt = f"<{'qQ' if is64 else 'iI'}"
            dyn_entry_sz = struct.calcsize(dyn_entry_fmt)
            soname_off = None
            strtab_off = None
            f.seek(dyn_off)
            for _ in range(dyn_sz // dyn_entry_sz):
                tag, val = struct.unpack(dyn_entry_fmt, f.read(dyn_entry_sz))
                if tag == 14:  # DT_SONAME
                    soname_off = val
                elif tag == 5:  # DT_STRTAB
                    strtab_off = val
                if soname_off is not None and strtab_off is not None:
                    break
            if soname_off is None or strtab_off is None:
                return None
            # strtab_off is a virtual address; find file offset via program headers
            strtab_file_off = None
            for i in range(e_phnum):
                f.seek(e_phoff + i * e_phentsize)
                p_type = struct.unpack("<I", f.read(4))[0]
                if p_type == 1:  # PT_LOAD
                    if is64:
                        f.seek(e_phoff + i * e_phentsize + 8)
                        p_offset = struct.unpack("<Q", f.read(8))[0]
                        p_vaddr = struct.unpack("<Q", f.read(8))[0]
                        f.read(8)  # p_paddr
                        p_filesz = struct.unpack("<Q", f.read(8))[0]
                    else:
                        f.seek(e_phoff + i * e_phentsize + 4)
                        p_offset = struct.unpack("<I", f.read(4))[0]
                        p_vaddr = struct.unpack("<I", f.read(4))[0]
                        f.read(4)  # p_paddr
                        p_filesz = struct.unpack("<I", f.read(4))[0]
                    if p_vaddr <= strtab_off < p_vaddr + p_filesz:
                        strtab_file_off = strtab_off - p_vaddr + p_offset
                        break
            if strtab_file_off is None:
                return None
            f.seek(strtab_file_off + soname_off)
            name = b""
            while True:
                c = f.read(1)
                if not c or c == b"\x00":
                    break
                name += c
            return name.decode("utf-8") if name else None
    except Exception:
        return None


def _create_soname_symlinks(pkg_dir: Path) -> int:
    """Create versioned soname symlinks for packaged .so files."""
    created = 0
    for so_file in sorted(pkg_dir.glob("*.so")):
        if so_file.is_symlink():
            continue
        soname = _read_elf_soname(so_file)
        if not soname or soname == so_file.name:
            continue
        # Sanity: soname must look like a library filename (lib*.so.*)
        if not soname.startswith("lib") or ".so." not in soname:
            continue
        link = pkg_dir / soname
        if not link.exists():
            link.symlink_to(so_file.name)
            log("LINK", f"{soname} -> {so_file.name}")
            created += 1
    return created


def _dest_path(src: FileSource, file_path: Path, root_path: Path | None = None) -> Path:
    if src.destination_subdir:
        if root_path is not None:
            return Path(src.destination_subdir) / file_path.relative_to(root_path)
        return Path(src.destination_subdir) / file_path.name
    if root_path is not None:
        return file_path.relative_to(root_path)
    return Path(file_path.name)


def collect(src: FileSource, ws: Path, cfg: str) -> tuple[list[tuple[Path, Path]], list[str]]:
    path = src.resolve(ws, cfg)
    ok_sfx = {s.lower() for s in src.suffixes}

    if src.source_kind == "file":
        if not path.exists():
            return [], [f"{src.name}: not found: {path}"]
        if not path.is_file():
            return [], [f"{src.name}: not a file: {path}"]
        return [(path, _dest_path(src, path))], []

    if src.source_kind == "tree":
        if not path.is_dir():
            return [], [f"{src.name}: directory not found: {path}"]
        files = sorted(
            f for f in path.rglob("*")
            if f.is_file()
            and f.suffix.lower() in ok_sfx
            and (src.name_filter is None or f.name in src.name_filter)
        )
        if not files:
            return [], [f"{src.name}: no matching files in {path}"]
        return [(f, _dest_path(src, f, path)) for f in files], []

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
    return [(f, _dest_path(src, f)) for f in files], []


def copy_file(src: Path, dst_dir: Path, relative_dst: Path) -> tuple[str, int]:
    dst = dst_dir / relative_dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    sz = src.stat().st_size

    if dst.exists() and dst.is_file():
        if filecmp.cmp(src, dst, shallow=False):
            log("SKIP", f"{relative_dst} (identical, {fmt(sz)})")
            return "skipped", 0
        log("COPY", f"{relative_dst} -> overwrite ({fmt(sz)})")
        shutil.copy2(src, dst)
        return "overwritten", sz

    log("COPY", f"{relative_dst} ({fmt(sz)})")
    shutil.copy2(src, dst)
    return "copied", sz


# ---------------------------------------------------------------------------
# IR generation
# ---------------------------------------------------------------------------

CONVERT_IR_EXE = "convert_ir.exe" if IS_WINDOWS else "convert_ir"


def _needs_ir_generation(model_dir: Path, group_size: int | None, vl: bool) -> bool:
    """Return True if the expected text IR files are missing from model_dir."""
    gs_tag = f"_g{group_size}" if group_size is not None else ""
    vl_tag = "_vl" if vl else ""
    # Look for qwen3_5_text[_vl]_*[_gN].xml
    for f in model_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name.startswith("qwen3_5_text") and name.endswith(".xml"):
            if vl_tag and vl_tag not in name:
                continue
            if gs_tag and gs_tag not in name:
                continue
            return False
    return True


def _find_convert_ir(ws: Path, cfg: str) -> Path | None:
    """Locate convert_ir executable in the build tree."""
    candidates = [
        ws / "openvino.genai" / "build" / "bin" / cfg / CONVERT_IR_EXE,
        ws / "openvino.genai" / "build" / "bin" / CONVERT_IR_EXE,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _find_runtime_dirs(ws: Path, cfg: str) -> list[Path]:
    """Return runtime library directories needed for convert_ir."""
    dirs = []
    ov_bin = ws / "openvino" / "bin" / "intel64" / cfg
    if ov_bin.is_dir():
        dirs.append(ov_bin)
    tbb_dir = ws / TBB_RELATIVE
    if tbb_dir.is_dir():
        dirs.append(tbb_dir)
    genai_lib = ws / "openvino.genai" / "build" / "openvino_genai"
    if genai_lib.is_dir():
        dirs.append(genai_lib)
    return dirs


def generate_ir(ws: Path, cfg: str, model_dir: Path, group_size: int | None,
                quant_mode: str, backup_mode: str, vl: bool) -> bool:
    """Run convert_ir to generate model IR files. Returns True on success."""
    exe = _find_convert_ir(ws, cfg)
    if exe is None:
        log("ERROR", f"convert_ir executable not found in build tree")
        return False

    cmd = [str(exe), "--model", str(model_dir), "--force"]
    if vl:
        cmd.append("--vl")

    env = os.environ.copy()
    env["OV_GENAI_USE_MODELING_API"] = "1"
    env["OV_GENAI_INFLIGHT_QUANT_MODE"] = quant_mode
    if group_size is not None:
        env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = str(group_size)
    env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = backup_mode

    # Add runtime dirs to PATH/LD_LIBRARY_PATH
    runtime_dirs = _find_runtime_dirs(ws, cfg)
    if IS_WINDOWS:
        path_var = "PATH"
    else:
        path_var = "LD_LIBRARY_PATH"
    existing = env.get(path_var, "")
    prepend = os.pathsep.join(str(d) for d in runtime_dirs)
    env[path_var] = prepend + os.pathsep + existing if existing else prepend

    log("INFO", f"Generating IR: {' '.join(cmd)}")
    log("INFO", f"  quant_mode={quant_mode}, backup_mode={backup_mode}, group_size={group_size}")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        if result.returncode != 0:
            log("ERROR", f"convert_ir failed with exit code {result.returncode}")
            return False
        log("INFO", "IR generation completed successfully")
        return True
    except subprocess.TimeoutExpired:
        log("ERROR", "convert_ir timed out (600s)")
        return False
    except Exception as e:
        log("ERROR", f"Failed to run convert_ir: {e}")
        return False


def collect_model_files(model_dir: Path, model_subdir: str, include_hf_weights: bool,
                        group_size: int | None = None) -> tuple[list[tuple[Path, Path]], list[str]]:
    if not model_dir.exists():
        return [], [f"Model directory not found: {model_dir}"]
    if not model_dir.is_dir():
        return [], [f"Model path is not a directory: {model_dir}"]

    # Build group-size tag for filtering IR files (e.g. "_g128.")
    gs_tag = f"_g{group_size}." if group_size is not None else None

    matched: list[tuple[Path, Path]] = []
    for file_path in sorted(model_dir.iterdir()):
        if not file_path.is_file():
            continue
        # Filter IR files by group size when specified
        if file_path.suffix.lower() in {".xml", ".bin"}:
            stem = file_path.stem + file_path.suffix  # full filename
            if gs_tag is not None and file_path.name.startswith("qwen3_5_text"):
                if gs_tag not in stem:
                    log("SKIP", f"{file_path.name} (group_size != {group_size})")
                    continue
            matched.append((file_path, Path("models") / model_subdir / file_path.name))
        elif (
            file_path.name in MODEL_METADATA_FILES
            or (include_hf_weights and file_path.suffix.lower() == ".safetensors")
            or (include_hf_weights and file_path.name.endswith(".safetensors.index.json"))
        ):
            matched.append((file_path, Path("models") / model_subdir / file_path.name))

    if not matched:
        return [], [f"No model files selected from {model_dir}"]

    names = {path.name for path, _ in matched}
    has_vl_text_xml = any(name.startswith("qwen3_5_text_vl") and name.endswith(".xml") for name in names)
    has_vl_text_bin = any(name.startswith("qwen3_5_text_vl") and name.endswith(".bin") for name in names)
    if has_vl_text_xml and has_vl_text_bin:
        filtered: list[tuple[Path, Path]] = []
        for file_path, relative_dst in matched:
            if (
                file_path.suffix.lower() in {".xml", ".bin"}
                and file_path.name.startswith("qwen3_5_text")
                and not file_path.name.startswith("qwen3_5_text_vl")
            ):
                log("SKIP", f"{relative_dst} (redundant when qwen3_5_text_vl_* is present)")
                continue
            filtered.append((file_path, relative_dst))
        matched = filtered

    names = {path.name for path, _ in matched}
    issues: list[str] = []
    if "config.json" not in names:
        issues.append(f"Model package is missing required file: {model_dir / 'config.json'}")
    if not any(name.startswith("qwen3_5_text") and name.endswith(".xml") for name in names):
        issues.append(f"Model package is missing text IR XML files in {model_dir}")
    if not any(name.startswith("qwen3_5_text") and name.endswith(".bin") for name in names):
        issues.append(f"Model package is missing text IR BIN files in {model_dir}")
    if issues:
        return [], issues
    return matched, []


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
            "  python scripts\\package_serve.py --model-dir /models/Qwen3.5-9B\n"
            "  python scripts\\package_serve.py --include-tokenizer-python\n"
        ),
    )
    p.add_argument("--clean", action="store_true",
                   help="Remove existing files in destination before copying.")
    p.add_argument("--output", type=str, default=None, metavar="DIR",
                   help="Override output root (default: <workspace>/package_serve).")
    p.add_argument("--build-type", choices=("Release", "RelWithDebInfo"),
                   default="Release", help="Build configuration (default: Release).")
    p.add_argument("--model-dir", type=str, default=None, metavar="DIR",
                   help="Optional model directory to bundle under models/<dirname>.")
    p.add_argument("--include-hf-weights", action="store_true",
                   help="Also bundle HuggingFace .safetensors weights. Default is IR-only packaging.")
    p.add_argument("--group-size", type=int, default=128, metavar="N",
                   help="Only package text IR files matching this group size (default: 128).\n"
                        "Set to 0 to include all group sizes.")
    p.add_argument("--quant-mode", type=str, default="int4_asym",
                   choices=["int4_sym", "int4_asym", "int8_sym", "int8_asym"],
                   help="Quantization mode for IR generation (default: int4_asym).")
    p.add_argument("--backup-mode", type=str, default="int4_sym",
                   choices=["int4_sym", "int4_asym", "int8_sym", "int8_asym"],
                   help="Backup quantization mode (default: int4_sym).")
    p.add_argument("--no-vl", action="store_true",
                   help="Skip vision-language IR generation (VL is enabled by default).")
    p.add_argument("--include-tokenizer-python", action="store_true",
                   help="Bundle Python openvino/openvino_tokenizers fallback packages for runtime tokenizer conversion.")
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
    log("INFO", f"Tokenizer Python fallback: {'enabled' if args.include_tokenizer_python else 'disabled'}")

    model_dir = Path(args.model_dir).resolve() if args.model_dir else None
    model_subdir = model_dir.name if model_dir else None
    group_size = args.group_size if args.group_size != 0 else None
    if model_dir is not None:
        log("INFO", f"Model src : {model_dir}")
        log("INFO", f"Model dst : {pkg_dir / 'models' / model_subdir}")
        log("INFO", f"Group size: {group_size if group_size else 'all'}")

    stats = dict(matched=0, copied=0, overwritten=0, skipped=0, errors=0,
                 matched_bytes=0, written_bytes=0, cleaned=0, cleaned_bytes=0)
    sources = BASE_SOURCES + (TOKENIZER_FALLBACK_SOURCES if args.include_tokenizer_python else ())

    # Clean
    if args.clean:
        for f in pkg_dir.iterdir():
            if f.is_file() and (f.suffix.lower() in PACKAGE_CLEAN_SUFFIXES or f.name in SERVE_EXECUTABLES):
                sz = f.stat().st_size
                log("CLEAN", f"{f.name} ({fmt(sz)})")
                f.unlink()
                stats["cleaned"] += 1
                stats["cleaned_bytes"] += sz
        for src in ALL_SOURCES:
            if src.destination_subdir:
                subtree = pkg_dir / src.destination_subdir
                if subtree.exists():
                    log("CLEAN", f"{subtree.relative_to(pkg_dir)}/")
                    shutil.rmtree(subtree)
        models_dir = pkg_dir / "models"
        if models_dir.exists():
            log("CLEAN", f"{models_dir.relative_to(pkg_dir)}/")
            shutil.rmtree(models_dir)
        log("INFO", f"Cleaned {stats['cleaned']} file(s), {fmt(stats['cleaned_bytes'])}")

    # Collect and copy
    for src in sources:
        files, issues = collect(src, ws, cfg)
        if issues:
            for iss in issues:
                log("ERROR", iss)
            stats["errors"] += len(issues)
            continue

        log("INFO", f"{src.name}: {len(files)} file(s)")
        for f, rel_dst in files:
            stats["matched"] += 1
            stats["matched_bytes"] += f.stat().st_size
            try:
                action, written = copy_file(f, pkg_dir, rel_dst)
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

    if model_dir is not None and model_subdir is not None:
        # Regenerate IR: always when --clean, otherwise only if missing
        vl = not args.no_vl
        if args.clean or _needs_ir_generation(model_dir, group_size, vl):
            reason = "forced by --clean" if args.clean else "text IR not found"
            log("INFO", f"Generating IR ({reason})...")
            if not generate_ir(ws, cfg, model_dir, group_size, args.quant_mode, args.backup_mode, vl):
                log("ERROR", "IR generation failed, cannot bundle model")
                return 1
        files, issues = collect_model_files(model_dir, model_subdir, args.include_hf_weights, group_size)
        if issues:
            for iss in issues:
                log("ERROR", iss)
            stats["errors"] += len(issues)
        else:
            log("INFO", f"Bundled model files: {len(files)} file(s)")
            for f, rel_dst in files:
                stats["matched"] += 1
                stats["matched_bytes"] += f.stat().st_size
                try:
                    action, written = copy_file(f, pkg_dir, rel_dst)
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

    # Copy readme.txt to package root
    readme_src = Path(__file__).resolve().parent.parent / "serving" / "readme.txt"
    if readme_src.is_file():
        shutil.copy2(readme_src, pkg_dir / "readme.txt")
        log("INFO", "Copied readme.txt to package root")

    # Create versioned soname symlinks (Linux only)
    if not IS_WINDOWS:
        n_links = _create_soname_symlinks(pkg_dir)
        if n_links:
            log("INFO", f"Created {n_links} soname symlink(s)")

    # Patch RUNPATH to $ORIGIN so binaries and libs find each other locally
    if not IS_WINDOWS:
        patchelf_ok = True
        targets = [f for f in pkg_dir.glob("*") if f.is_file() and not f.is_symlink()
                    and (f.name in SERVE_EXECUTABLES or f.suffix == ".so")]
        for target in sorted(targets):
            try:
                result = subprocess.run(
                    ["patchelf", "--print-rpath", str(target)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0 or "$ORIGIN" in result.stdout:
                    continue
                rpath = result.stdout.strip()
                if not rpath:
                    continue
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN", str(target)],
                    check=True, capture_output=True, text=True,
                )
                log("PATCH", f"{target.name}: RUNPATH -> $ORIGIN")
            except FileNotFoundError:
                log("WARN", "patchelf not found, skipping RUNPATH patch")
                patchelf_ok = False
                break
            except subprocess.CalledProcessError as e:
                log("WARN", f"patchelf failed on {target.name}: {e.stderr.strip()}")

    # Summary
    final_files = [f for f in pkg_dir.rglob("*") if f.is_file()]
    final_bytes = sum(f.stat().st_size for f in final_files)
    executables = sorted(f.name for f in final_files if f.name in SERVE_EXECUTABLES)
    runtime_libs = [f.name for f in final_files if f.suffix.lower() in {".dll", ".so", ".12", ".2"}]

    log("SUMMARY", "=" * 60)
    log("SUMMARY", f"Destination  : {pkg_dir}")
    log("SUMMARY", f"Matched      : {stats['matched']} files ({fmt(stats['matched_bytes'])})")
    log("SUMMARY", f"Copied       : {stats['copied']}, overwritten: {stats['overwritten']}, skipped: {stats['skipped']}")
    log("SUMMARY", f"Errors       : {stats['errors']}")
    log("SUMMARY", f"Written      : {fmt(stats['written_bytes'])}")
    log("SUMMARY", f"Package total: {len(final_files)} files ({fmt(final_bytes)})")
    log("SUMMARY", f"Executables  : {', '.join(executables)}")
    log("SUMMARY", f"Runtime libs : {len(runtime_libs)}")
    if model_subdir is not None:
        log("SUMMARY", f"Bundled model: models/{model_subdir}")
    log("SUMMARY", "=" * 60)

    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
