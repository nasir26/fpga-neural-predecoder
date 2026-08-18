# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Parses the Vitis HLS csynth XML report and records latency/II/resource
numbers into results/metrics/hls_{precision}.json. Never hand-transcribe a
synthesis number into the paper; this script is the single source.
"""
from __future__ import annotations

import json
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


def main() -> None:
    precision = sys.argv[1] if len(sys.argv) > 1 else "int8"
    xml_path = REPO_ROOT / "hls" / "predecoder_proj" / "solution1" / "syn" / "report" / "predecoder_kernel_csynth.xml"
    if not xml_path.exists():
        raise SystemExit(f"csynth report not found: {xml_path}")

    metrics = parse_csynth_xml(xml_path)
    metrics["precision"] = precision
    metrics["source_xml"] = str(xml_path)

    out_path = REPO_ROOT / "results" / "metrics" / f"hls_{precision}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
