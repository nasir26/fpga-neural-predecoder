# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Parses a Vitis HLS csynth report (XML from the vitis_hls project flow, or
the text .rpt emitted by the `v++ -c -t hw` command-line flow, which does not
produce an XML report at all) and records latency/II/resource numbers into
results/metrics/hls_{precision}.json. Never hand-transcribe a synthesis
number into the paper; this script is the single source.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_csynth_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()

    def text(path: str, default=None):
        el = root.find(path)
        return el.text if el is not None else default

    resources = {
        "BRAM_18K": int(text(".//AreaEstimates/Resources/BRAM_18K", 0)),
        "DSP": int(text(".//AreaEstimates/Resources/DSP", 0)),
        "FF": int(text(".//AreaEstimates/Resources/FF", 0)),
        "LUT": int(text(".//AreaEstimates/Resources/LUT", 0)),
        "URAM": int(text(".//AreaEstimates/Resources/URAM", 0)),
    }
    available = {
        "BRAM_18K": int(text(".//AreaEstimates/AvailableResources/BRAM_18K", 0)),
        "DSP": int(text(".//AreaEstimates/AvailableResources/DSP", 0)),
        "FF": int(text(".//AreaEstimates/AvailableResources/FF", 0)),
        "LUT": int(text(".//AreaEstimates/AvailableResources/LUT", 0)),
        "URAM": int(text(".//AreaEstimates/AvailableResources/URAM", 0)),
    }
    utilization_pct = {
        k: (100.0 * resources[k] / available[k] if available[k] > 0 else None) for k in resources
    }

    return {
        "target_device": text(".//UserAssignments/Part"),
        "target_clock_period_ns": float(text(".//UserAssignments/TargetClockPeriod", 0)),
        "estimated_clock_period_ns": float(text(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod", 0)),
        "latency_best_case_cycles": int(text(".//SummaryOfOverallLatency/Best-caseLatency", 0)),
        "latency_avg_case_cycles": int(text(".//SummaryOfOverallLatency/Average-caseLatency", 0)),
        "latency_worst_case_cycles": int(text(".//SummaryOfOverallLatency/Worst-caseLatency", 0)),
        "latency_best_case_ms": text(".//SummaryOfOverallLatency/Best-caseRealTimeLatency"),
        "latency_worst_case_ms": text(".//SummaryOfOverallLatency/Worst-caseRealTimeLatency"),
        "dataflow_interval_min_cycles": int(text(".//DataflowPipelineThroughput/range/min", 0)),
        "dataflow_interval_max_cycles": int(text(".//DataflowPipelineThroughput/range/max", 0)),
        "resources_used": resources,
        "resources_available": available,
        "utilization_pct": utilization_pct,
    }


def parse_csynth_rpt(rpt_path: Path) -> dict:
    content = rpt_path.read_text()

    def require(pattern: str, scope: str = content):
        m = re.search(pattern, scope)
        if not m:
            raise ValueError(f"pattern not found in {rpt_path}: {pattern!r}")
        return m

    target_device = require(r"Target device:\s*(\S+)").group(1)

    timing = require(r"\|ap_clk\s*\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|")
    target_clock_period_ns = float(timing.group(1))
    estimated_clock_period_ns = float(timing.group(2))

    perf_scope = content[content.index("== Performance Estimates") : content.index("== Utilization Estimates")]
    latency = require(
        r"\|\s*(\d+)\|\s*(\d+)\|\s*([\d.]+ \w+)\|\s*([\d.]+ \w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\w+)\|",
        perf_scope,
    )

    util_start = content.index("== Utilization Estimates")
    util_scope = content[util_start : content.index("+ Detail:", util_start)]
    resource_keys = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    total_row = require(r"\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", util_scope)
    available_row = require(r"\|Available\s+\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", util_scope)

    resources = {k: int(v) for k, v in zip(resource_keys, total_row.groups())}
    available = {k: int(v) for k, v in zip(resource_keys, available_row.groups())}
    utilization_pct = {
        k: (100.0 * resources[k] / available[k] if available[k] > 0 else None) for k in resources
    }

    return {
        "target_device": target_device,
        "target_clock_period_ns": target_clock_period_ns,
        "estimated_clock_period_ns": estimated_clock_period_ns,
        "latency_best_case_cycles": int(latency.group(1)),
        "latency_avg_case_cycles": int(latency.group(1)),
        "latency_worst_case_cycles": int(latency.group(2)),
        "latency_best_case_ms": latency.group(3),
        "latency_worst_case_ms": latency.group(4),
        "dataflow_interval_min_cycles": int(latency.group(5)),
        "dataflow_interval_max_cycles": int(latency.group(6)),
        "resources_used": resources,
        "resources_available": available,
        "utilization_pct": utilization_pct,
    }


def main() -> None:
    precision = sys.argv[1] if len(sys.argv) > 1 else "int8"
    source_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if source_arg is not None:
        source_path = Path(source_arg)
    else:
        source_path = REPO_ROOT / "hls" / "predecoder_proj" / "solution1" / "syn" / "report" / "predecoder_kernel_csynth.xml"
    if not source_path.exists():
        raise SystemExit(f"csynth report not found: {source_path}")

    if source_path.suffix == ".rpt":
        metrics = parse_csynth_rpt(source_path)
    else:
        metrics = parse_csynth_xml(source_path)
    metrics["precision"] = precision
    metrics["source_xml"] = str(source_path)

    out_path = REPO_ROOT / "results" / "metrics" / f"hls_{precision}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
