# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Parses the real v++ -l -t hw (place-and-route / bitstream link) output
reports and records the actual, post-route build result into
results/metrics/hw_build_{precision}.json. Distinct from
results/metrics/hls_{precision}.json (the pre-route HLS csynth estimate,
see scripts/record_hls_metrics.py): this script records what the real
Vivado implementation run actually achieved (timing closure, kernel-scoped
post-route resource usage), not an estimate. Never hand-transcribe a
synthesis or implementation number; this script is the single source.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_timing_summary(rpt_path: Path) -> dict:
    content = rpt_path.read_text()
    m = re.search(
        r"\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+"
        r"(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s*\n",
        content,
    )
    if not m:
        raise ValueError(f"timing summary row not found in {rpt_path}")
    return {
        "wns_ns": float(m.group(1)),
        "tns_ns": float(m.group(2)),
        "tns_failing_endpoints": int(m.group(3)),
        "tns_total_endpoints": int(m.group(4)),
        "whs_ns": float(m.group(5)),
        "ths_ns": float(m.group(6)),
        "ths_failing_endpoints": int(m.group(7)),
        "ths_total_endpoints": int(m.group(8)),
        "timing_closed": float(m.group(1)) >= 0.0 and int(m.group(3)) == 0,
    }


def parse_kernel_utilization(rpt_path: Path, kernel_instance: str) -> dict:
    content = rpt_path.read_text()
    m = re.search(
        r"\|\s*" + re.escape(kernel_instance) + r"\s*\|"
        r"\s*(\d+)\s*\[\s*([\d.]+)%\]\s*\|"
        r"\s*(\d+)\s*\[\s*([\d.]+)%\]\s*\|"
        r"\s*(\d+)\s*\[\s*([\d.]+)%\]\s*\|"
        r"\s*(\d+)\s*\[\s*([\d.]+)%\]\s*\|"
        r"\s*(\d+)\s*\[\s*([\d.]+)%\]\s*\|"
        r"\s*(\d+)\s*\[\s*([\d.]+)%\]\s*\|",
        content,
    )
    if not m:
        raise ValueError(f"kernel row '{kernel_instance}' not found in {rpt_path}")
    g = m.groups()
    return {
        "LUT": {"used": int(g[0]), "pct_of_user_budget": float(g[1])},
        "LUTAsMem": {"used": int(g[2]), "pct_of_user_budget": float(g[3])},
        "FF": {"used": int(g[4]), "pct_of_user_budget": float(g[5])},
        "BRAM": {"used": int(g[6]), "pct_of_user_budget": float(g[7])},
        "URAM": {"used": int(g[8]), "pct_of_user_budget": float(g[9])},
        "DSP": {"used": int(g[10]), "pct_of_user_budget": float(g[11])},
    }


def main() -> None:
    precision = sys.argv[1] if len(sys.argv) > 1 else "int8"
    link_reports = REPO_ROOT / "hls" / "_x" / "reports" / "link" / "imp"
    timing_rpt = link_reports / "impl_1_hw_bb_locked_timing_summary_routed.rpt"
    util_rpt = link_reports / "impl_1_kernel_util_routed.rpt"
    xclbin_path = REPO_ROOT / "hls" / "build_hw" / "predecoder_kernel.xclbin"
    hls_metrics_path = REPO_ROOT / "results" / "metrics" / f"hls_{precision}.json"

    for p in (timing_rpt, util_rpt, xclbin_path, hls_metrics_path):
        if not p.exists():
            raise SystemExit(f"required file not found: {p}")

    hls_metrics = json.loads(hls_metrics_path.read_text())
    timing = parse_timing_summary(timing_rpt)
    kernel_util = parse_kernel_utilization(util_rpt, "predecoder_kernel_1")

    kernel_clock_hz = 100_000_000
    worst_cycles = hls_metrics["latency_worst_case_cycles"]
    best_cycles = hls_metrics["latency_best_case_cycles"]

    metrics = {
        "precision": precision,
        "xclbin_path": str(xclbin_path.relative_to(REPO_ROOT)),
        "xclbin_size_bytes": xclbin_path.stat().st_size,
        "kernel_clock_hz": kernel_clock_hz,
        "timing": timing,
        "kernel_utilization_post_route": kernel_util,
        "latency_worst_case_cycles": worst_cycles,
        "latency_best_case_cycles": best_cycles,
        "latency_worst_case_ms_at_kernel_clock": worst_cycles / kernel_clock_hz * 1000,
        "latency_best_case_ms_at_kernel_clock": best_cycles / kernel_clock_hz * 1000,
        "source_timing_rpt": str(timing_rpt.relative_to(REPO_ROOT)),
        "source_util_rpt": str(util_rpt.relative_to(REPO_ROOT)),
        "note": (
            "kernel_utilization_post_route is scoped to predecoder_kernel_1 only "
            "(excludes the U55C platform shell), from Vivado's post-route "
            "report_accelerator_utilization, not an HLS estimate. Percentages are "
            "of the kernel's User Budget (device resources minus what the "
            "platform shell reserves), not of total device resources -- not "
            "directly comparable to results/metrics/hls_{precision}.json's "
            "device-wide percentages."
        ),
    }

    out_path = REPO_ROOT / "results" / "metrics" / f"hw_build_{precision}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
