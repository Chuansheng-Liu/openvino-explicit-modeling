# GPUView ETL File Analysis for GPU Performance

**Date:** 2026-03-30
**Hardware:** Intel Arc 140T iGPU, Windows 11
**Input:** `OV_8B.etl` (raw GPU trace, no ETW markers)

---

## Background

GPUView captures ETL files containing `Microsoft-Windows-DxgKrnl` DMA packet events.
Without custom ETW markers, analysis is limited to GPU scheduling data:
when each engine was busy, how long each dispatch took, and how large the idle gaps
between dispatches are. This is sufficient to characterize GPU utilization, dispatch
overhead, and compute vs copy engine split for an OpenVINO inference run.

Note: the trace includes GPU activity from **all processes** (display compositor, browser,
etc.) unless filtered by PID. Recommended practice: close other GPU-intensive apps before
tracing, and optionally use `--pid` to restrict analysis to the OpenVINO process.

---

## Architecture

```
OV_8B.etl
    ↓ tracerpt.exe -of CSV (subprocess, one-time)
<basename>_events.csv  (temp, stream-parsed, deleted after use)
    ↓ Python: filter DmaPacket/Start + Stop, pair by sequence number
list of packets: (start_ms, stop_ms, duration_ms, engine)
    ↓ metrics computation (utilization, percentiles, gaps, histogram)
stats dict
    ↓ HTML template (inline Chart.js CDN + SVG timeline)
report.html
```

---

## Section 1: Script Interface

**File:** `scripts/analyze_etl.py`

```
python analyze_etl.py --etl <path/to/file.etl> [--out report.html] [--pid <PID>] [--no-convert]
```

- `--etl`: path to the ETL file (required)
- `--out`: output HTML file path (default: `<etl_dir>/<basename>_report.html`)
- `--pid`: filter DMA packets to this process ID only (optional; default: all processes)
- `--no-convert`: skip tracerpt conversion; expect `<etl_dir>/<basename>_events.csv` to already exist

The temp CSV is always named `<etl_dir>/<basename>_events.csv` (same basename as the ETL,
`_events.csv` suffix). The `--no-convert` flag looks for exactly this filename.

---

## Section 2: ETL → CSV Conversion

Uses `tracerpt.exe` (ships with Windows, no extra install).

```python
subprocess.run([
    "tracerpt", etl_path,
    "-of", "CSV",
    "-o", tmp_csv,
    "-lr",   # lenient: skip malformed/truncated records at end of trace
    "-y",    # overwrite without prompt
], check=True)
```

**Error handling:**
- If `tracerpt.exe` is not found: print a clear message directing the user to run on Windows
  with the Windows Performance Toolkit installed. Exit with code 1.
- If `tracerpt.exe` returns non-zero: print the stderr output and exit with code 1.
- If the output CSV is empty or missing after conversion: print error and exit with code 1.

---

## Section 3: Event Parsing

### CSV header detection

Read the first row as the header. Build a `col` dict: `col["FieldName"] = index`.
All subsequent field lookups use `col["FieldName"]` — no hardcoded column indices.

On first run (or when debugging), print the detected headers so the user can validate
column names against the actual tracerpt output on their Windows version:

```
[INFO] tracerpt CSV headers (19 columns): EventName, TimeStamp, PID, TID, ...
```

### Timestamp parsing

tracerpt CSV emits timestamps as formatted strings, not raw ticks. The format on
Windows 11 is: `MM/DD/YYYY-HH:MM:SS.mmmuuunnn`
(month/day/year-hour:min:sec.milliseconds-microseconds-nanoseconds, no separators
between sub-second units in some locales). Parse with:

```python
from datetime import datetime

def parse_ts_ms(s: str) -> float:
    # Format: "03/30/2026-14:23:01.123456789"
    # Split on '-' to get date, time+subsec
    # Parse date+time up to '.', then handle sub-second part
    dt_part, subsec = s.rsplit(".", 1)
    dt = datetime.strptime(dt_part, "%m/%d/%Y-%H:%M:%S")
    # subsec is 9 digits: mmm uuu nnn (milliseconds packed)
    subsec_ms = int(subsec.ljust(9, "0")[:9]) / 1_000_000
    return dt.timestamp() * 1000 + subsec_ms
```

Normalize all timestamps so the first DMA event = t=0 ms.

**Fallback:** If timestamp parsing fails (locale uses a different format), the script
prints the raw timestamp value from the first event and exits with a clear message:
`"Unrecognized timestamp format: '<value>'. Please open an issue with this value."`.

### Target event names

Filter rows where `EventName` column matches (case-insensitive prefix):
- `"DmaPacket/Start"` or `"Microsoft-Windows-DxgKrnl/DmaPacket/Start"`
- `"DmaPacket/Stop"` or `"Microsoft-Windows-DxgKrnl/DmaPacket/Stop"`

If zero matching rows are found after parsing the full CSV, print:
`"No DmaPacket events found. Verify the ETL was captured with DxgKrnl provider (log.cmd or DiagEasy)."` and exit 1.

### Packet pairing

Match `DmaPacket/Start` to `DmaPacket/Stop` by a sequence number field.
The field name varies by Windows version; try in order:
1. `"SubmitSequence"`
2. `"ulQueueSubmitSequence"`
3. `"PacketSequence"`
4. Fallback: match by insertion order (FIFO per context handle)

Unmatched starts (truncated at end of trace) are silently discarded.

### Engine classification

tracerpt CSV field names for DxgKrnl events vary. The engine type is derived from
the `PacketType` field (if present) or inferred from the context handle:

```python
def classify_engine(row, col):
    if "PacketType" in col:
        pt = int(row[col["PacketType"]])
        if pt == 2:   return "Compute"
        if pt == 3:   return "Copy"
        return "Other"
    # Fallback: if PacketType column is absent, classify all as "Compute"
    # (single-engine traces from simple DxgKrnl captures)
    return "Compute"
```

Print a one-time warning if `PacketType` is not found:
`"[WARN] PacketType column not found — all packets classified as Compute."`.

### PID filtering

If `--pid` is given, skip rows where `PID` column != the specified PID.

### Output

A list of `Packet(start_ms, stop_ms, duration_ms, engine)` sorted by `start_ms`.
Collect into an in-memory list (size bounded by packet count, not CSV size — typically
tens of thousands of packets, not millions).

Duration list is collected per engine for percentile computation (stdlib `statistics.quantiles`).

---

## Section 4: Metrics Computation

### Overlapping interval handling

Before computing `busy_ms`, merge overlapping packets per engine using a sweep:

```python
def merge_intervals(packets):
    """Returns list of (start, stop) with overlaps merged."""
    sorted_pkts = sorted(packets, key=lambda p: p.start_ms)
    merged = []
    for p in sorted_pkts:
        if merged and p.start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], p.stop_ms))
        else:
            merged.append((p.start_ms, p.stop_ms))
    return merged
```

`busy_ms = sum(stop - start for start, stop in merged_intervals)`

### Utilization definition

`utilization_pct = busy_ms / gpu_active_duration_ms * 100`

Where `gpu_active_duration_ms` = timestamp of last DMA event − timestamp of first DMA event.
This is the GPU-active portion of the trace (excludes pre-trace idle lead-in).
**This is by design**: utilization reflects GPU activity density during the run, not
capture wall-clock time.

### Idle gaps

Computed from merged intervals (not raw packets) to avoid negative gaps from overlaps:

```python
gaps = [merged[i+1][0] - merged[i][1] for i in range(len(merged)-1)]
gaps = [g for g in gaps if g > 0]  # discard zero or negative (shouldn't occur after merge)
```

### Metrics table (per engine + overall)

| Metric | Description |
|--------|-------------|
| `total_trace_ms` | GPU-active duration (first→last DMA event) |
| `busy_ms` | Sum of merged busy intervals |
| `utilization_pct` | `busy_ms / total_trace_ms * 100` |
| `packet_count` | Raw (pre-merge) packet count |
| `avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `max_ms` | Percentiles on raw duration list |
| `avg_gap_ms`, `max_gap_ms` | From merged idle gaps |
| `idle_pct` | `100 - utilization_pct` |
| `duration_histogram` | 20 log-scale bins from min→max duration |

---

## Section 5: HTML Report Structure

Single standalone HTML file. Chart.js loaded from CDN — requires internet to view charts.
If the machine is air-gapped, the report still renders tables and SVG timeline; only the
Chart.js bar charts will be blank (graceful degradation — no error, just empty chart areas).

**Sections:**

1. **Summary cards** (4 cards):
   - Total trace duration (ms)
   - Compute engine utilization %
   - Copy engine utilization %
   - Total packet count

2. **Engine breakdown table**:
   - Rows: Compute, Copy, Other
   - Columns: Count, Total ms, Avg ms, P95 ms, Max ms, Utilization %

3. **GPU activity timeline** (SVG, no JS required):
   - X axis = time (full GPU-active duration)
   - Two rows: Compute engine, Copy engine
   - Colored bars: compute=blue (`#4e79a7`), copy=orange (`#f28e2b`), idle=light gray
   - Downsampling: if packet count > 5000, group packets into fixed-width time buckets
     (bucket_width = total_duration / 5000). For each bucket, render a single bar from
     `min(start)` to `max(stop)` of all packets in that bucket. This preserves the
     busy/idle pattern without per-pixel granularity beyond what the SVG can show.
   - `viewBox="0 0 1000 80"` with `preserveAspectRatio="none"` for responsive width

4. **Duration distribution chart** (Chart.js bar):
   - Log-scale X axis (20 duration bins)
   - Y axis = packet count per bin
   - Compute and Copy as separate grouped bar datasets

5. **Idle gap distribution** (Chart.js bar):
   - Same log-scale bin approach as duration distribution
   - Shows gap between consecutive GPU dispatches

6. **Top-20 longest packets** (table):
   - Rank, Engine, Start (ms), Duration (ms)

---

## Section 6: Error Handling Summary

| Condition | Behavior |
|-----------|----------|
| `tracerpt.exe` not found | Print: "tracerpt.exe not found. Run on Windows with WPT installed." Exit 1. |
| `tracerpt.exe` exits non-zero | Print stderr. Exit 1. |
| CSV empty or missing after conversion | Print error. Exit 1. |
| Zero DmaPacket events in CSV | Print: "No DmaPacket events found. Verify DxgKrnl provider." Exit 1. |
| Timestamp format unrecognized | Print raw value. Ask user to report. Exit 1. |
| `PacketType` column missing | Warn once, classify all as Compute. Continue. |
| Output directory doesn't exist | Create it (with `mkdir -p`). |
| Output file not writable | Print error. Exit 1. |

---

## Success Criteria

The script is working correctly when:
1. It exits with code 0
2. The HTML file is created at the specified output path
3. Opening the HTML in a browser shows all 6 sections with non-zero data
4. Compute utilization % is between 0 and 100 (merged intervals, no overflow)
5. The SVG timeline shows visible activity bars (not all-gray)

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/analyze_etl.py` | Main analysis script |
| `<etl_dir>/<basename>_events.csv` | Temp tracerpt CSV output (deleted after parse) |
| `<etl_dir>/<basename>_report.html` | Generated report (default output location) |

---

## Test Command

```
python scripts/analyze_etl.py \
  --etl "C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview\OV_8B.etl"
```

Output: `C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\gpuview\OV_8B_report.html`
