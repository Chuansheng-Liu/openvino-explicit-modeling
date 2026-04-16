# GPUView ETL Analysis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/analyze_etl.py` — a CLI tool that converts a GPUView ETL file to a standalone HTML performance report showing GPU utilization, engine breakdown, activity timeline, duration histograms, and top-N longest packets.

**Architecture:** `tracerpt.exe` converts the ETL to CSV (subprocess); Python stream-parses the CSV for `DmaPacket/Start`+`Stop` events; metrics are computed from merged intervals; a single-file HTML report with inline Chart.js and SVG is generated.

**Tech Stack:** Python 3.8+ stdlib only (`subprocess`, `csv`, `statistics`, `math`, `json`, `pathlib`, `argparse`, `datetime`). Chart.js 4.x via CDN. `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-03-30-gpuview-etl-analysis-design.md`

---

> **Working directory for all commands:** `D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling`
> Run every `python -m pytest` and `git` command from this directory.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/analyze_etl.py` | Create | Main script: CLI, conversion, parsing, metrics, HTML |
| `scripts/tests/test_analyze_etl.py` | Create | Unit tests for parsing, metrics, HTML generation |

All logic lives in one file (`analyze_etl.py`) — the task is small enough that splitting into modules would be over-engineering. Functions are independently testable.

---

## Chunk 1: Skeleton, CLI, and ETL→CSV Conversion

### Task 1: Script skeleton + CLI

**Files:**
- Create: `scripts/analyze_etl.py`
- Create: `scripts/tests/test_analyze_etl.py`

- [ ] **Step 1: Create the test file**

```python
# scripts/tests/test_analyze_etl.py
from __future__ import annotations
import pytest
from pathlib import Path


def test_placeholder():
    """Placeholder — replaced in subsequent tasks."""
    assert True
```

- [ ] **Step 2: Create the script skeleton with CLI**

```python
# scripts/analyze_etl.py
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Packet(NamedTuple):
    start_ms: float
    stop_ms: float
    duration_ms: float
    engine: str   # "Compute", "Copy", "Other"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze a GPUView ETL file and produce an HTML performance report."
    )
    p.add_argument("--etl", required=True, type=Path,
                   help="Path to the .etl file")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML path (default: <etl_dir>/<basename>_report.html)")
    p.add_argument("--pid", type=int, default=None,
                   help="Filter DMA packets to this process ID (default: all processes)")
    p.add_argument("--no-convert", action="store_true",
                   help="Skip tracerpt conversion; use existing <basename>_events.csv")
    return p.parse_args(argv)


def default_out_path(etl_path: Path) -> Path:
    return etl_path.parent / (etl_path.stem + "_report.html")


def default_csv_path(etl_path: Path) -> Path:
    return etl_path.parent / (etl_path.stem + "_events.csv")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    etl_path = args.etl.resolve()
    out_path = args.out.resolve() if args.out else default_out_path(etl_path)
    csv_path = default_csv_path(etl_path)

    if not etl_path.exists():
        print(f"ERROR: ETL file not found: {etl_path}", file=sys.stderr)
        return 1

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_convert:
        rc = convert_etl_to_csv(etl_path, csv_path)
        if rc != 0:
            return rc

    result = parse_csv(csv_path, pid_filter=args.pid)
    if result is None:
        return 1  # fatal parse error already printed (e.g. bad timestamp format)
    packets = result
    if not packets:
        print(
            "ERROR: No DmaPacket events found. "
            "Verify the ETL was captured with DxgKrnl provider (log.cmd or DiagEasy).",
            file=sys.stderr,
        )
        return 1

    stats = compute_stats(packets)
    html = build_html(stats, packets, etl_path.name)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to: {out_path}")

    # Clean up temp CSV unless --no-convert was requested
    if not args.no_convert and csv_path.exists():
        csv_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run placeholder test**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v
```

Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add analyze_etl.py skeleton + CLI"
```

---

### Task 2: ETL → CSV conversion

**Files:**
- Modify: `scripts/analyze_etl.py` — add `convert_etl_to_csv()`
- Modify: `scripts/tests/test_analyze_etl.py` — add conversion error tests

- [ ] **Step 1: Write failing tests for conversion errors**

```python
# scripts/tests/test_analyze_etl.py
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add scripts dir to path so we can import analyze_etl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_etl


def test_no_convert_skips_tracerpt(tmp_path):
    """--no-convert flag: convert_etl_to_csv is never called."""
    etl = tmp_path / "test.etl"
    etl.touch()
    # Pre-create a minimal CSV so parse_csv has something to read
    csv_file = tmp_path / "test_events.csv"
    csv_file.write_text(
        "EventName,TimeStamp,PID,TID,SubmitSequence,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,1,2\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.010000000,100,200,1,2\n",
        encoding="utf-8",
    )
    out_html = tmp_path / "test_report.html"
    with patch.object(analyze_etl, "convert_etl_to_csv") as mock_convert:
        rc = analyze_etl.main(["--etl", str(etl), "--out", str(out_html), "--no-convert"])
    mock_convert.assert_not_called()
    assert rc == 0


def test_convert_missing_tracerpt(tmp_path):
    """convert_etl_to_csv returns 1 and prints error if tracerpt not found."""
    etl = tmp_path / "test.etl"
    etl.touch()
    csv_out = tmp_path / "test_events.csv"
    with patch("subprocess.run", side_effect=FileNotFoundError("tracerpt not found")):
        rc = analyze_etl.convert_etl_to_csv(etl, csv_out)
    assert rc == 1


def test_convert_nonzero_exit(tmp_path, capsys):
    """convert_etl_to_csv returns 1 if tracerpt exits non-zero."""
    etl = tmp_path / "test.etl"
    etl.touch()
    csv_out = tmp_path / "test_events.csv"
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "some error"
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "tracerpt", stderr="bad")):
        rc = analyze_etl.convert_etl_to_csv(etl, csv_out)
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v
```

Expected: `2 failed` (function not yet defined)

- [ ] **Step 3: Implement `convert_etl_to_csv()`**

Add to `scripts/analyze_etl.py`:

```python
def convert_etl_to_csv(etl_path: Path, csv_path: Path) -> int:
    """Run tracerpt to convert ETL to CSV. Returns 0 on success, 1 on failure."""
    try:
        subprocess.run(
            [
                "tracerpt", str(etl_path),
                "-of", "CSV",
                "-o", str(csv_path),
                "-lr",
                "-y",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "ERROR: tracerpt.exe not found. "
            "Run on Windows with the Windows Performance Toolkit installed.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print(f"ERROR: tracerpt failed:\n{e.stderr}", file=sys.stderr)
        return 1

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("ERROR: tracerpt produced no output CSV.", file=sys.stderr)
        return 1

    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add ETL->CSV conversion with error handling"
```

---

## Chunk 2: CSV Parsing — Timestamps and Packet Pairing

### Task 3: Timestamp parsing

**Files:**
- Modify: `scripts/analyze_etl.py` — add `parse_ts_ms()`
- Modify: `scripts/tests/test_analyze_etl.py` — add timestamp tests

- [ ] **Step 1: Write failing tests**

```python
def test_parse_ts_ms_basic():
    """Parse standard tracerpt timestamp format."""
    ts = "03/30/2026-14:23:01.123456789"
    ms = analyze_etl.parse_ts_ms(ts)
    # Should be a float (exact value depends on timezone, but it should parse)
    assert isinstance(ms, float)
    assert ms > 0


def test_parse_ts_ms_subsecond_precision():
    """Sub-second part is parsed to millisecond resolution."""
    ts1 = "03/30/2026-14:23:01.000000000"
    ts2 = "03/30/2026-14:23:01.001000000"
    ms1 = analyze_etl.parse_ts_ms(ts1)
    ms2 = analyze_etl.parse_ts_ms(ts2)
    assert abs((ms2 - ms1) - 1.0) < 0.001  # 1 ms apart


def test_parse_ts_ms_bad_format(capsys):
    """Returns None and prints error on unrecognized format."""
    result = analyze_etl.parse_ts_ms("not-a-timestamp")
    assert result is None
    captured = capsys.readouterr()
    assert "Unrecognized timestamp format" in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py::test_parse_ts_ms_basic -v
```

Expected: `FAILED` (function not defined)

- [ ] **Step 3: Implement `parse_ts_ms()`**

Add to `scripts/analyze_etl.py`:

```python
def parse_ts_ms(s: str) -> Optional[float]:
    """Parse a tracerpt CSV timestamp string to milliseconds.

    Expected format: MM/DD/YYYY-HH:MM:SS.mmmuuunnn
    Returns None and prints an error on unrecognized format.
    """
    try:
        dt_part, subsec = s.rsplit(".", 1)
        dt = datetime.strptime(dt_part, "%m/%d/%Y-%H:%M:%S")
        subsec_ms = int(subsec.ljust(9, "0")[:9]) / 1_000_000
        return dt.timestamp() * 1000.0 + subsec_ms
    except (ValueError, AttributeError):
        print(
            f"ERROR: Unrecognized timestamp format: '{s}'. "
            "Please report this value.",
            file=sys.stderr,
        )
        return None
```

- [ ] **Step 4: Run tests**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "ts_ms"
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add timestamp parser for tracerpt CSV format"
```

---

### Task 4: CSV parsing — header detection, packet pairing, engine classification

**Files:**
- Modify: `scripts/analyze_etl.py` — add `parse_csv()`
- Modify: `scripts/tests/test_analyze_etl.py` — add CSV parsing tests

- [ ] **Step 1: Write failing tests**

```python
def test_parse_csv_basic(tmp_path):
    """Parses a minimal CSV with one matched DmaPacket pair."""
    csv_text = (
        "EventName,TimeStamp,PID,TID,SubmitSequence,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,1,2\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.010000000,100,200,1,2\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    packets = analyze_etl.parse_csv(csv_file, pid_filter=None)
    assert len(packets) == 1
    assert packets[0].engine == "Compute"
    assert abs(packets[0].duration_ms - 10.0) < 0.1


def test_parse_csv_pid_filter(tmp_path):
    """PID filter keeps only matching process packets."""
    csv_text = (
        "EventName,TimeStamp,PID,TID,SubmitSequence,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,1,2\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.010000000,100,200,1,2\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,999,200,2,2\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.010000000,999,200,2,2\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    packets = analyze_etl.parse_csv(csv_file, pid_filter=100)
    assert len(packets) == 1


def test_parse_csv_copy_engine(tmp_path):
    """PacketType 3 is classified as Copy engine."""
    csv_text = (
        "EventName,TimeStamp,PID,TID,SubmitSequence,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,1,3\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.005000000,100,200,1,3\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    packets = analyze_etl.parse_csv(csv_file, pid_filter=None)
    assert packets[0].engine == "Copy"


def test_parse_csv_no_packet_type_column(tmp_path, capsys):
    """Missing PacketType column: warns once, classifies all as Compute."""
    csv_text = (
        "EventName,TimeStamp,PID,TID,SubmitSequence\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,1\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.010000000,100,200,1\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    packets = analyze_etl.parse_csv(csv_file, pid_filter=None)
    assert len(packets) == 1
    assert packets[0].engine == "Compute"
    captured = capsys.readouterr()
    assert "PacketType" in captured.err


def test_parse_csv_unmatched_start_discarded(tmp_path):
    """Start without matching Stop is silently discarded."""
    csv_text = (
        "EventName,TimeStamp,PID,TID,SubmitSequence,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,1,2\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    packets = analyze_etl.parse_csv(csv_file, pid_filter=None)
    assert packets == []


def test_parse_csv_fifo_fallback(tmp_path):
    """When no sequence field exists, pairs packets FIFO."""
    # CSV with no SubmitSequence / ulQueueSubmitSequence / PacketSequence
    csv_text = (
        "EventName,TimeStamp,PID,TID,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,100,200,2\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.008000000,100,200,2\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    packets = analyze_etl.parse_csv(csv_file, pid_filter=None)
    assert len(packets) == 1
    assert abs(packets[0].duration_ms - 8.0) < 0.1


def test_parse_csv_pid_filter_missing_pid_col(tmp_path):
    """PID filter active but PID column absent: all packets dropped (silent, no crash)."""
    csv_text = (
        "EventName,TimeStamp,SubmitSequence,PacketType\n"
        "DmaPacket/Start,03/30/2026-00:00:00.000000000,1,2\n"
        "DmaPacket/Stop,03/30/2026-00:00:00.010000000,1,2\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    # With pid_filter set but PID column missing, parse_csv should not crash
    # and should return [] (all rows fail the PID check with IndexError -> continue)
    packets = analyze_etl.parse_csv(csv_file, pid_filter=100)
    assert packets == []


def test_parse_csv_bad_timestamp_returns_none(tmp_path, capsys):
    """Bad timestamp in a matched event: parse_csv returns None (not empty list)."""
    csv_text = (
        "EventName,TimeStamp,PID,TID,SubmitSequence,PacketType\n"
        "DmaPacket/Start,NOT-A-TIMESTAMP,100,200,1,2\n"
        "DmaPacket/Stop,NOT-A-TIMESTAMP,100,200,1,2\n"
    )
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    result = analyze_etl.parse_csv(csv_file, pid_filter=None)
    assert result is None  # None signals fatal error, not "no events"
    captured = capsys.readouterr()
    assert "Unrecognized timestamp format" in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "parse_csv"
```

Expected: all `FAILED`

- [ ] **Step 3: Implement `parse_csv()`**

Add to `scripts/analyze_etl.py`:

```python
# Sequence field names to try in order
_SEQ_FIELD_CANDIDATES = [
    "SubmitSequence",
    "ulQueueSubmitSequence",
    "PacketSequence",
]

_START_SUFFIXES = ("DmaPacket/Start",)
_STOP_SUFFIXES = ("DmaPacket/Stop",)


def _classify_engine(row: List[str], col: Dict[str, int]) -> str:
    if "PacketType" not in col:
        return "Compute"
    try:
        pt = int(row[col["PacketType"]])
    except (ValueError, IndexError):
        return "Other"
    if pt == 2:
        return "Compute"
    if pt == 3:
        return "Copy"
    return "Other"


def parse_csv(csv_path: Path, pid_filter: Optional[int]) -> Optional[List[Packet]]:
    """Stream-parse a tracerpt CSV file and return matched DMA packets.

    Returns None on a fatal parse error (e.g. unrecognized timestamp format).
    Returns [] if the file parsed cleanly but contained no DmaPacket events.
    """
    packets: List[Packet] = []
    # pending_starts: seq_key -> (start_ms, engine)
    pending: Dict[str, Tuple[float, str]] = {}
    warned_packet_type = False
    t0: Optional[float] = None

    with csv_path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)

        # Detect headers
        try:
            header = next(reader)
        except StopIteration:
            return packets
        col = {name.strip(): idx for idx, name in enumerate(header)}

        print(f"[INFO] tracerpt CSV headers ({len(header)} columns): {', '.join(header)}")

        # Determine sequence field
        seq_field = next((s for s in _SEQ_FIELD_CANDIDATES if s in col), None)

        if "PacketType" not in col and not warned_packet_type:
            print(
                "[WARN] PacketType column not found — all packets classified as Compute.",
                file=sys.stderr,
            )
            warned_packet_type = True

        event_col = col.get("EventName", 0)
        ts_col = col.get("TimeStamp", 1)
        pid_col = col.get("PID", 2)

        fifo_starts: List[Tuple[float, str]] = []  # fallback FIFO queue

        for row in reader:
            if len(row) <= max(event_col, ts_col):
                continue

            event = row[event_col].strip()
            # Use endswith only — covers both "DmaPacket/Start" and
            # "Microsoft-Windows-DxgKrnl/DmaPacket/Start" with one check
            is_start = any(event.endswith(s) for s in _START_SUFFIXES)
            is_stop = any(event.endswith(s) for s in _STOP_SUFFIXES)

            if not (is_start or is_stop):
                continue

            # PID filter
            if pid_filter is not None:
                try:
                    if int(row[pid_col].strip()) != pid_filter:
                        continue
                except (ValueError, IndexError):
                    continue

            ts = parse_ts_ms(row[ts_col].strip())
            if ts is None:
                return None  # fatal: timestamp format error already printed

            if t0 is None:
                t0 = ts
            ts_norm = ts - t0

            engine = _classify_engine(row, col)

            if seq_field is not None:
                try:
                    seq = row[col[seq_field]].strip()
                except IndexError:
                    seq = ""
                key = f"{engine}:{seq}"
            else:
                key = None

            if is_start:
                if key is not None:
                    pending[key] = (ts_norm, engine)
                else:
                    fifo_starts.append((ts_norm, engine))
            elif is_stop:
                if key is not None:
                    if key in pending:
                        start_ms, eng = pending.pop(key)
                        dur = ts_norm - start_ms
                        if dur >= 0:
                            packets.append(Packet(start_ms, ts_norm, dur, eng))
                else:
                    if fifo_starts:
                        start_ms, eng = fifo_starts.pop(0)
                        dur = ts_norm - start_ms
                        if dur >= 0:
                            packets.append(Packet(start_ms, ts_norm, dur, eng))

    packets.sort(key=lambda p: p.start_ms)
    return packets
```

- [ ] **Step 4: Run tests**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "parse_csv"
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add CSV parser — header detection, packet pairing, engine classification"
```

---

## Chunk 3: Metrics Computation

### Task 5: Metrics computation

**Files:**
- Modify: `scripts/analyze_etl.py` — add `merge_intervals()`, `compute_stats()`
- Modify: `scripts/tests/test_analyze_etl.py` — add stats tests

- [ ] **Step 1: Write failing tests**

```python
def _make_packets(specs: list) -> List[analyze_etl.Packet]:
    """Helper: list of (start_ms, stop_ms, engine) tuples -> Packet list."""
    return [
        analyze_etl.Packet(s, e, e - s, eng)
        for s, e, eng in specs
    ]


def test_merge_intervals_no_overlap():
    intervals = [(0.0, 5.0), (10.0, 15.0)]
    merged = analyze_etl.merge_intervals(intervals)
    assert merged == [(0.0, 5.0), (10.0, 15.0)]


def test_merge_intervals_overlap():
    intervals = [(0.0, 10.0), (5.0, 15.0)]
    merged = analyze_etl.merge_intervals(intervals)
    assert merged == [(0.0, 15.0)]


def test_merge_intervals_contained():
    intervals = [(0.0, 20.0), (5.0, 10.0)]
    merged = analyze_etl.merge_intervals(intervals)
    assert merged == [(0.0, 20.0)]


def test_compute_stats_basic():
    packets = _make_packets([
        (0.0, 10.0, "Compute"),
        (20.0, 25.0, "Compute"),
        (5.0, 8.0, "Copy"),
    ])
    stats = analyze_etl.compute_stats(packets)
    assert stats["total_trace_ms"] == pytest.approx(25.0, abs=0.01)
    assert stats["engines"]["Compute"]["packet_count"] == 2
    assert stats["engines"]["Compute"]["busy_ms"] == pytest.approx(15.0, abs=0.01)
    # utilization: 15/25 = 60%
    assert stats["engines"]["Compute"]["utilization_pct"] == pytest.approx(60.0, abs=0.1)


def test_compute_stats_overlapping_packets():
    """Overlapping compute packets: busy_ms uses merged intervals, not sum."""
    packets = _make_packets([
        (0.0, 10.0, "Compute"),
        (5.0, 15.0, "Compute"),  # overlaps with first
    ])
    stats = analyze_etl.compute_stats(packets)
    # merged = [(0,15)], busy_ms = 15, not 20
    assert stats["engines"]["Compute"]["busy_ms"] == pytest.approx(15.0, abs=0.01)
    assert stats["engines"]["Compute"]["utilization_pct"] <= 100.0


def test_compute_stats_idle_gaps():
    packets = _make_packets([
        (0.0, 5.0, "Compute"),
        (10.0, 15.0, "Compute"),   # gap of 5ms
        (20.0, 25.0, "Compute"),   # gap of 5ms
    ])
    stats = analyze_etl.compute_stats(packets)
    eng = stats["engines"]["Compute"]
    assert eng["avg_gap_ms"] == pytest.approx(5.0, abs=0.01)
    assert eng["max_gap_ms"] == pytest.approx(5.0, abs=0.01)


def test_compute_stats_empty():
    """compute_stats with empty list returns empty dict (not a crash)."""
    stats = analyze_etl.compute_stats([])
    assert stats == {}


def test_make_histogram_label_format():
    """Histogram bins have 'x.xxx–y.yyy' label format and zero-count bins omitted."""
    values = [1.0, 10.0, 100.0]
    hist = analyze_etl._make_histogram(values, n_bins=5)
    assert len(hist) > 0
    for bin_ in hist:
        assert "label" in bin_
        assert "count" in bin_
        assert "–" in bin_["label"]          # separator between low and high
        assert bin_["count"] > 0              # zero-count bins are omitted


def test_make_histogram_zero_count_bins_omitted():
    """Bins with zero count are not included in the output."""
    # All values are identical → only one bin has any count
    values = [5.0] * 10
    hist = analyze_etl._make_histogram(values, n_bins=20)
    # With min==max, function returns a single-bin summary
    assert len(hist) == 1
    assert hist[0]["count"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "merge_intervals or compute_stats"
```

Expected: all `FAILED`

- [ ] **Step 3: Implement `merge_intervals()` and `compute_stats()`**

Add to `scripts/analyze_etl.py`:

```python
def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping (start, stop) intervals. Input need not be sorted."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_ivs[0]]
    for start, stop in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _make_histogram(values: List[float], n_bins: int = 20) -> List[Dict]:
    """Log-scale histogram. Returns list of {label, count}."""
    if not values:
        return []
    min_v = max(min(values), 1e-6)
    max_v = max(values)
    if min_v >= max_v:
        return [{"label": f"{min_v:.3f}", "count": len(values)}]
    log_min = math.log10(min_v)
    log_max = math.log10(max_v)
    edges = [10 ** (log_min + i * (log_max - log_min) / n_bins) for i in range(n_bins + 1)]
    bins = [0] * n_bins
    for v in values:
        for i in range(n_bins):
            if v <= edges[i + 1]:
                bins[i] += 1
                break
        else:
            bins[-1] += 1
    return [
        {"label": f"{edges[i]:.3f}–{edges[i+1]:.3f}", "count": bins[i]}
        for i in range(n_bins)
        if bins[i] > 0
    ]


def compute_stats(packets: List[Packet]) -> Dict:
    """Compute per-engine and overall performance metrics."""
    if not packets:
        return {}

    total_trace_ms = packets[-1].stop_ms - packets[0].start_ms

    engines = {}
    for eng in ("Compute", "Copy", "Other"):
        eng_packets = [p for p in packets if p.engine == eng]
        if not eng_packets:
            continue

        intervals = [(p.start_ms, p.stop_ms) for p in eng_packets]
        merged = merge_intervals(intervals)
        busy_ms = sum(stop - start for start, stop in merged)
        utilization_pct = (busy_ms / total_trace_ms * 100) if total_trace_ms > 0 else 0.0

        durations = [p.duration_ms for p in eng_packets]
        gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        gaps = [g for g in gaps if g > 0]

        try:
            quantiles = statistics.quantiles(durations, n=100) if len(durations) >= 4 else []
            p50 = quantiles[49] if quantiles else (durations[0] if durations else 0.0)
            p95 = quantiles[94] if quantiles else (durations[0] if durations else 0.0)
            p99 = quantiles[98] if quantiles else (durations[0] if durations else 0.0)
        except statistics.StatisticsError:
            p50 = p95 = p99 = durations[0] if durations else 0.0

        engines[eng] = {
            "packet_count": len(eng_packets),
            "busy_ms": busy_ms,
            "utilization_pct": utilization_pct,
            "avg_ms": statistics.mean(durations),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "max_ms": max(durations),
            "avg_gap_ms": statistics.mean(gaps) if gaps else 0.0,
            "max_gap_ms": max(gaps) if gaps else 0.0,
            "idle_pct": 100.0 - utilization_pct,
            "duration_histogram": _make_histogram(durations),
            "gap_histogram": _make_histogram(gaps),
        }

    total_packets = len(packets)
    top20 = sorted(packets, key=lambda p: p.duration_ms, reverse=True)[:20]

    return {
        "total_trace_ms": total_trace_ms,
        "total_packet_count": total_packets,
        "engines": engines,
        "top20_longest": [
            {"rank": i + 1, "engine": p.engine,
             "start_ms": round(p.start_ms, 3), "duration_ms": round(p.duration_ms, 3)}
            for i, p in enumerate(top20)
        ],
        "all_packets_for_timeline": [
            {"start": p.start_ms, "stop": p.stop_ms, "engine": p.engine}
            for p in packets
        ],
    }
```

- [ ] **Step 4: Run tests**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "merge_intervals or compute_stats or histogram"
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add metrics computation — merge intervals, percentiles, histograms"
```

---

## Chunk 4: HTML Report Generation

### Task 6: HTML report — summary, engine table, top-20 table

**Files:**
- Modify: `scripts/analyze_etl.py` — add `build_html()`, `_render_summary_cards()`, `_render_engine_table()`, `_render_top20_table()`
- Modify: `scripts/tests/test_analyze_etl.py` — add HTML content tests

- [ ] **Step 1: Write failing tests**

```python
def _sample_stats():
    packets = _make_packets([
        (0.0, 10.0, "Compute"),
        (15.0, 20.0, "Compute"),
        (3.0, 6.0, "Copy"),
    ])
    return analyze_etl.compute_stats(packets)


def test_build_html_produces_html(tmp_path):
    """build_html returns a string starting with <!DOCTYPE html>."""
    html = analyze_etl.build_html(_sample_stats(), [], "test.etl")
    assert html.strip().startswith("<!DOCTYPE html>")


def test_build_html_contains_summary_cards():
    html = analyze_etl.build_html(_sample_stats(), [], "test.etl")
    assert "Total Duration" in html
    assert "Compute Utilization" in html


def test_build_html_contains_engine_table():
    html = analyze_etl.build_html(_sample_stats(), [], "test.etl")
    assert "Compute" in html
    assert "Copy" in html
    assert "packet_count" not in html  # key names should not leak into HTML


def test_build_html_contains_top20_table():
    stats = _sample_stats()
    html = analyze_etl.build_html(stats, [], "test.etl")
    assert "Top-20 Longest Packets" in html
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "build_html"
```

Expected: all `FAILED`

- [ ] **Step 3: Implement `build_html()` with summary, table, top-20**

Add to `scripts/analyze_etl.py`:

```python
def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def _render_summary_cards(stats: Dict) -> str:
    engines = stats.get("engines", {})
    compute_util = _fmt(engines.get("Compute", {}).get("utilization_pct", 0.0))
    copy_util = _fmt(engines.get("Copy", {}).get("utilization_pct", 0.0))
    cards = [
        ("Total Duration", f"{_fmt(stats['total_trace_ms'])} ms"),
        ("Compute Utilization", f"{compute_util}%"),
        ("Copy Utilization", f"{copy_util}%"),
        ("Total Packets", str(stats["total_packet_count"])),
    ]
    items = "".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
        for label, value in cards
    )
    return f'<div class="cards">{items}</div>'


def _render_engine_table(stats: Dict) -> str:
    rows = ""
    for eng, d in stats.get("engines", {}).items():
        rows += (
            f"<tr><td>{eng}</td>"
            f"<td>{d['packet_count']}</td>"
            f"<td>{_fmt(d['busy_ms'])}</td>"
            f"<td>{_fmt(d['avg_ms'])}</td>"
            f"<td>{_fmt(d['p95_ms'])}</td>"
            f"<td>{_fmt(d['max_ms'])}</td>"
            f"<td>{_fmt(d['utilization_pct'])}%</td></tr>"
        )
    return (
        "<h2>Engine Breakdown</h2>"
        '<table><thead><tr>'
        "<th>Engine</th><th>Count</th><th>Total ms</th>"
        "<th>Avg ms</th><th>P95 ms</th><th>Max ms</th><th>Utilization</th>"
        "</tr></thead><tbody>"
        f"{rows}"
        "</tbody></table>"
    )


def _render_top20_table(stats: Dict) -> str:
    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['engine']}</td>"
        f"<td>{_fmt(r['start_ms'])}</td><td>{_fmt(r['duration_ms'])}</td></tr>"
        for r in stats.get("top20_longest", [])
    )
    return (
        "<h2>Top-20 Longest Packets</h2>"
        '<table><thead><tr>'
        "<th>#</th><th>Engine</th><th>Start (ms)</th><th>Duration (ms)</th>"
        "</tr></thead><tbody>"
        f"{rows}"
        "</tbody></table>"
    )


def build_html(stats: Dict, _packets_unused: list, etl_name: str) -> str:
    """Assemble the full HTML report. _packets_unused kept for API compat."""
    summary = _render_summary_cards(stats)
    engine_table = _render_engine_table(stats)
    timeline_svg = _render_timeline_svg(stats)
    charts_js = _render_charts(stats)
    top20 = _render_top20_table(stats)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GPU ETL Report: {etl_name}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
h1 {{ color: #222; }}
h2 {{ margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px 24px; min-width: 160px;
         box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
.card-label {{ font-size: 0.8em; color: #888; }}
.card-value {{ font-size: 1.8em; font-weight: bold; color: #333; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #f0f0f0; text-align: center; }}
td:first-child {{ text-align: left; }}
.chart-container {{ background: #fff; padding: 16px; border-radius: 8px;
                    box-shadow: 0 1px 4px rgba(0,0,0,.12); max-width: 800px; }}
svg.timeline {{ width: 100%; height: 80px; background: #f9f9f9;
               border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>GPU ETL Report: {etl_name}</h1>
{summary}
{engine_table}
<h2>GPU Activity Timeline</h2>
{timeline_svg}
{charts_js}
{top20}
</body>
</html>"""
```

- [ ] **Step 4: Add stubs for `_render_timeline_svg()` and `_render_charts()` so HTML builds**

```python
def _render_timeline_svg(stats: Dict) -> str:
    return "<!-- timeline: implemented in Task 7 -->"


def _render_charts(stats: Dict) -> str:
    return "<!-- charts: implemented in Task 8 -->"
```

- [ ] **Step 5: Run tests**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v -k "build_html"
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add HTML report skeleton — summary cards, engine table, top-20"
```

---

### Task 7: SVG Activity Timeline

**Files:**
- Modify: `scripts/analyze_etl.py` — replace `_render_timeline_svg()` stub

- [ ] **Step 1: Write failing test**

```python
def test_render_timeline_svg_contains_rects():
    """SVG timeline contains rect elements for packet bars."""
    stats = _sample_stats()
    svg = analyze_etl.build_html(stats, [], "t.etl")
    assert "<rect" in svg
    assert "viewBox" in svg
```

- [ ] **Step 2: Run test to verify it fails**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py::test_render_timeline_svg_contains_rects -v
```

Expected: `FAILED` (stub returns comment, no rect)

- [ ] **Step 3: Implement `_render_timeline_svg()`**

Replace the stub in `scripts/analyze_etl.py`:

```python
# Engine display config: (engine_name, y_offset_in_80px_viewbox, color)
_ENGINE_ROWS = [
    ("Compute", 5,  40, "#4e79a7"),  # (name, y, height, color)
    ("Copy",    50, 20, "#f28e2b"),
]

def _render_timeline_svg(stats: Dict) -> str:
    total_ms = stats.get("total_trace_ms", 1.0)
    if total_ms <= 0:
        total_ms = 1.0

    all_pkts = stats.get("all_packets_for_timeline", [])
    MAX_BARS = 5000
    VIEW_W = 1000.0

    def _packet_bars(engine: str, y: int, h: int, color: str) -> str:
        pkts = [p for p in all_pkts if p["engine"] == engine]
        if not pkts:
            return ""

        # Downsample if needed: group into fixed-width time buckets
        if len(pkts) > MAX_BARS:
            bucket_ms = total_ms / MAX_BARS
            buckets: Dict[int, Tuple[float, float]] = {}
            for p in pkts:
                b = int(p["start"] / bucket_ms)
                if b not in buckets:
                    buckets[b] = (p["start"], p["stop"])
                else:
                    buckets[b] = (min(buckets[b][0], p["start"]),
                                  max(buckets[b][1], p["stop"]))
            bars = list(buckets.values())
        else:
            bars = [(p["start"], p["stop"]) for p in pkts]

        rects = []
        for start, stop in bars:
            x = start / total_ms * VIEW_W
            w = max((stop - start) / total_ms * VIEW_W, 0.5)  # min 0.5px width
            rects.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{h}" fill="{color}"/>')
        return "\n".join(rects)

    # Labels
    labels = "".join(
        f'<text x="2" y="{y + h // 2 + 5}" font-size="9" fill="#555">{name}</text>'
        for name, y, h, _ in _ENGINE_ROWS
    )

    bars = "".join(
        _packet_bars(name, y, h, color)
        for name, y, h, color in _ENGINE_ROWS
    )

    return (
        f'<svg class="timeline" viewBox="0 0 {int(VIEW_W)} 80" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        f'{labels}{bars}'
        f"</svg>"
    )
```

- [ ] **Step 4: Run test**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py::test_render_timeline_svg_contains_rects -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add SVG GPU activity timeline with downsample"
```

---

### Task 8: Chart.js duration histogram and gap distribution

**Files:**
- Modify: `scripts/analyze_etl.py` — replace `_render_charts()` stub

- [ ] **Step 1: Write failing test**

```python
def test_render_charts_contains_chartjs():
    """HTML report includes Chart.js CDN reference and canvas elements."""
    html = analyze_etl.build_html(_sample_stats(), [], "t.etl")
    assert "chart.js" in html.lower()
    assert "<canvas" in html
```

- [ ] **Step 2: Run test to verify it fails**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py::test_render_charts_contains_chartjs -v
```

Expected: `FAILED`

- [ ] **Step 3: Implement `_render_charts()`**

Replace the stub:

```python
def _render_charts(stats: Dict) -> str:
    engines = stats.get("engines", {})

    def _hist_datasets(hist_key: str) -> Tuple[List[str], List[Dict]]:
        """Build Chart.js labels + datasets from per-engine histograms."""
        # Collect all bin labels across engines
        all_labels: List[str] = []
        for d in engines.values():
            for b in d.get(hist_key, []):
                if b["label"] not in all_labels:
                    all_labels.append(b["label"])

        COLORS = {"Compute": "#4e79a7", "Copy": "#f28e2b", "Other": "#76b7b2"}
        datasets = []
        for eng, d in engines.items():
            hist = {b["label"]: b["count"] for b in d.get(hist_key, [])}
            datasets.append({
                "label": eng,
                "data": [hist.get(lbl, 0) for lbl in all_labels],
                "backgroundColor": COLORS.get(eng, "#aaa"),
            })
        return all_labels, datasets

    dur_labels, dur_datasets = _hist_datasets("duration_histogram")
    gap_labels, gap_datasets = _hist_datasets("gap_histogram")

    dur_data = json.dumps({"labels": dur_labels, "datasets": dur_datasets})
    gap_data = json.dumps({"labels": gap_labels, "datasets": gap_datasets})

    return f"""
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<h2>Duration Distribution (log-scale bins, ms)</h2>
<div class="chart-container"><canvas id="durChart"></canvas></div>
<h2>Idle Gap Distribution (log-scale bins, ms)</h2>
<div class="chart-container"><canvas id="gapChart"></canvas></div>
<script>
(function() {{
  const mkChart = (id, data) => {{
    const ctx = document.getElementById(id);
    if (!ctx) return;
    new Chart(ctx, {{
      type: 'bar',
      data: data,
      options: {{
        responsive: true,
        scales: {{ x: {{ stacked: false }}, y: {{ beginAtZero: true }} }},
        plugins: {{ legend: {{ position: 'top' }} }}
      }}
    }});
  }};
  mkChart('durChart', {dur_data});
  mkChart('gapChart', {gap_data});
}})();
</script>"""
```

- [ ] **Step 4: Run test**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py::test_render_charts_contains_chartjs -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add Chart.js duration and gap distribution charts"
```

---

## Chunk 5: Integration and Smoke Test

### Task 9: Integration smoke test against real ETL file

**Files:**
- Modify: `scripts/tests/test_analyze_etl.py` — add smoke test (skipped if ETL absent)

- [ ] **Step 1: Add smoke test**

```python
import os
import re

# Allow override via environment variable; fall back to the GPUView default location
_DEFAULT_ETL = r"C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview\OV_8B.etl"
REAL_ETL = Path(os.environ.get("TEST_ETL_PATH", _DEFAULT_ETL))

_EXPECTED_H2_SECTIONS = [
    "Engine Breakdown",
    "GPU Activity Timeline",
    "Duration Distribution",
    "Idle Gap Distribution",
    "Top-20 Longest Packets",
]

@pytest.mark.skipif(not REAL_ETL.exists(), reason="Real ETL file not present — set TEST_ETL_PATH to enable")
def test_smoke_real_etl(tmp_path):
    """Full pipeline against a real ETL file — verifies non-empty report and spec success criteria."""
    out_html = tmp_path / "OV_8B_report.html"
    rc = analyze_etl.main([
        "--etl", str(REAL_ETL),
        "--out", str(out_html),
    ])
    assert rc == 0, "analyze_etl.main should exit 0"
    assert out_html.exists(), "HTML report should be created"
    html = out_html.read_text(encoding="utf-8")

    # Spec success criteria 1-3: HTML exists and has all sections
    assert "<!DOCTYPE html>" in html
    for heading in _EXPECTED_H2_SECTIONS:
        assert heading in html, f"Missing section: {heading}"

    # Spec success criteria 4: Compute utilization is between 0 and 100%
    # Find the utilization value in the summary card
    util_match = re.search(r"Compute Utilization.*?(\d+\.\d+)%", html, re.DOTALL)
    if util_match:
        util = float(util_match.group(1))
        assert 0.0 <= util <= 100.0, f"Compute utilization out of range: {util}"

    # Spec success criteria 5: SVG timeline has visible bars
    assert "<rect" in html, "SVG timeline should contain rect elements"
```

- [ ] **Step 2: Run smoke test (requires real ETL + tracerpt)**

```
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py::test_smoke_real_etl -v -s
```

Expected: `PASSED` (or `SKIPPED` if ETL not present)

- [ ] **Step 3: Manual verification**
  - Open the generated HTML in a browser
  - Verify: all 5 sections visible (Engine Breakdown, GPU Activity Timeline, Duration Distribution, Idle Gap Distribution, Top-20 Longest Packets), compute utilization 0–100%, SVG bars visible, Chart.js charts render
  - If `PacketType` warning appears, note the actual column names printed by `[INFO] tracerpt CSV headers`
    and adjust engine classification if needed

- [ ] **Step 4: Final commit**

```bash
git add scripts/analyze_etl.py scripts/tests/test_analyze_etl.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" \
  -m "feat(etl): add smoke test; complete analyze_etl.py implementation"
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python scripts/analyze_etl.py --etl path/to/file.etl` | Run analysis, output to `<basename>_report.html` |
| `python scripts/analyze_etl.py --etl path/to/file.etl --no-convert` | Re-run analysis using existing `_events.csv` |
| `python scripts/analyze_etl.py --etl path/to/file.etl --pid 12345` | Filter to one process |
| `cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling && python -m pytest scripts/tests/test_analyze_etl.py -v` | Run all unit tests |
