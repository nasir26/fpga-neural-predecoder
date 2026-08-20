# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Phase 6: regenerates every paper/README figure from committed
results/metrics/*.json only. No network access, no retraining, no re-running
HLS/v++ -- if a number isn't already on disk, this script skips that figure
and says so on stdout rather than inventing or estimating it.

Usage:
    python scripts/make_figures.py [--out-dir results/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = REPO_ROOT / "results" / "metrics"

# Fixed categorical order, validated colorblind-safe (dataviz skill,
# scripts/validate_palette.js): blue/orange/aqua pass CVD separation both
# adjacent (4-way line/bar charts) and all-pairs (the Fig. 6 scatter, which
# only ever shows 3 hues at once -- fp32/baseline use neutral gray/black,
# not a 4th hue, so the scatter never needs the 4th slot that fails
# all-pairs). See docs/DECISIONS.md for the validator output.
COLOR = {"baseline": "#555555", "fp32": "#1a1a1a", "int8": "#2a78d6", "int4": "#eb6834", "ternary": "#1baf7a"}
MARKER = {"baseline": "x", "fp32": "*", "int8": "s", "int4": "^", "ternary": "D"}
LABEL = {"baseline": "PyMatching only", "fp32": "FP32", "int8": "INT8", "int4": "INT4", "ternary": "Ternary"}
PRECISIONS = ["fp32", "int8", "int4", "ternary"]
BIT_WIDTH = {"fp32": 32.0, "int8": 8.0, "int4": 4.0, "ternary": 1.585}  # ternary = log2(3)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  skip: {path.relative_to(REPO_ROOT)} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fig1_ler_vs_p(sweep: dict, out_dir: Path, distance: int = 5) -> None:
    """LER vs physical error rate p, at the training distance, one line per
    precision plus the PyMatching-only baseline -- the pseudothreshold-style
    view of whether the pre-decoder helps or hurts as noise increases.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    baseline_ps, baseline_lers = [], []
    for precision in PRECISIONS:
        rows = sorted(
            (r for r in sweep[precision]["results"] if r["distance"] == distance), key=lambda r: r["p"]
        )
        if not rows:
            continue
        ps = [r["p"] for r in rows]
        lers = [r["ler"] for r in rows]
        ax.plot(
            ps, lers, color=COLOR[precision], marker=MARKER[precision], markersize=5,
            linewidth=1.6, label=LABEL[precision],
        )
        if precision == "fp32":
            baseline_ps = ps
            baseline_lers = [r["baseline_ler"] for r in rows]

    if baseline_ps:
        ax.plot(
            baseline_ps, baseline_lers, color=COLOR["baseline"], marker=MARKER["baseline"],
            markersize=5, linewidth=1.4, linestyle="--", label=LABEL["baseline"],
        )

    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.set_xticks(baseline_ps)
    ax.set_xticklabels([f"{p:g}" for p in baseline_ps])
    ax.set_title(f"LER vs. $p$ at $d={distance}$ (training distance)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "fig1_ler_vs_p.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def fig2_ler_reduction_factor_vs_d(sweep: dict, out_dir: Path, p: float = 0.003) -> None:
    """ler_reduction_factor (baseline_ler / ler; >1 means the pre-decoder
    helped, <1 means it hurt) vs. distance at the canonical eval p used
    throughout the project. This is the figure that shows the real,
    honestly-reported finding: it gets worse, not better, past the R=5
    training distance (docs/LIMITATIONS.md, Phase 2).
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for precision in PRECISIONS:
        rows = sorted(
            (r for r in sweep[precision]["results"] if r["p"] == p), key=lambda r: r["distance"]
        )
        if not rows:
            continue
        ds = [r["distance"] for r in rows]
        factors = [r["ler_reduction_factor"] for r in rows]
        ax.plot(
            ds, factors, color=COLOR[precision], marker=MARKER[precision], markersize=5,
            linewidth=1.6, label=LABEL[precision],
        )
    ax.axhline(1.0, color=COLOR["baseline"], linestyle="--", linewidth=1.2, label="baseline (no change)")
    ax.set_xlabel("code distance $d$ (eval; training distance = 5)")
    ax.set_ylabel("LER reduction factor\n(baseline LER / pre-decoder LER)")
    ax.set_yscale("log")
    ax.set_title(f"LER reduction factor vs. $d$ at $p={p}$")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "fig2_ler_reduction_factor_vs_d.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def fig3_sdr_vs_d(sweep: dict, out_dir: Path, p: float = 0.003) -> None:
    """Syndrome density reduction vs. distance: even though LER does not
    improve past d=5 (Fig. 2), the pre-decoder keeps removing a real
    fraction of detection events at every distance tested -- the two
    findings are not the same claim, and both are reported.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for precision in PRECISIONS:
        rows = sorted(
            (r for r in sweep[precision]["results"] if r["p"] == p), key=lambda r: r["distance"]
        )
        if not rows:
            continue
        ds = [r["distance"] for r in rows]
        sdrs = [r["syndrome_density_reduction"] for r in rows]
        ax.plot(
            ds, sdrs, color=COLOR[precision], marker=MARKER[precision], markersize=5,
            linewidth=1.6, label=LABEL[precision],
        )
    ax.set_xlabel("code distance $d$")
    ax.set_ylabel("syndrome density reduction")
    ax.set_ylim(0, 1)
    ax.set_title(f"Syndrome density reduction vs. $d$ at $p={p}$")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "fig3_sdr_vs_d.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def fig4_latency_breakdown(out_dir: Path) -> None:
    """Stacked per-round latency breakdown across three real, measured
    pipelines: PyMatching alone (Phase 1 baseline), the CPU-simulated
    pre-decoder+PyMatching pipeline (Phase 5's honest golden-model
    fallback, host_run_d5_p0.003.json, since no real hardware run has
    happened), and the real synthesized-and-routed FPGA kernel's own cycle
    count (hw_build_int8.json) combined with the same CPU PyMatching decode
    time (the FPGA kernel replaces only the pre-decoder stage; decode still
    runs on the host in this design). Skips silently only if a required
    input file is missing.
    """
    baseline = load_json(METRICS / "baseline_d5_r5_z_p0.003.json")
    host_run = load_json(METRICS / "host_run_d5_p0.003.json")
    hw_build = load_json(METRICS / "hw_build_int8.json")
    if not (baseline and host_run and hw_build):
        print("  skip fig4: missing one or more required inputs")
        return

    distance = hw_build_distance = 5
    fpga_model_us_per_round = (
        hw_build["latency_worst_case_ms_at_kernel_clock"] * 1000.0 / hw_build_distance
    )
    decode_us_per_round = host_run["mean_latency_us_per_round"]["pymatching_decode"]

    # Not a stacked bar: the decode segment (~1-2us) is 1-3 orders of
    # magnitude smaller than the model/kernel segment, and stacking is an
    # additive operation that a log-scale y-axis renders misleadingly (a
    # first attempt at this figure did exactly that and made the smallest
    # segment invisible). Total latency per pipeline as single bars (log
    # scale is fine for un-stacked bars), with the model/decode split as
    # direct value labels instead of visual area-stacking.
    labels = [
        "PyMatching only\n(CPU baseline)",
        "Pre-decoder + PyMatching\n(CPU, simulated)",
        "Pre-decoder (FPGA, real,\n100MHz) + PyMatching (CPU)",
    ]
    model_us = [0.0, host_run["mean_latency_us_per_round"]["predecoder_model"], fpga_model_us_per_round]
    decode_us = [
        baseline["mean_latency_us_per_round"],
        host_run["mean_latency_us_per_round"]["pymatching_decode"],
        decode_us_per_round,
    ]
    totals = [m + d for m, d in zip(model_us, decode_us)]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = list(range(len(labels)))
    ax.bar(x, totals, color=[COLOR["baseline"], COLOR["int8"], COLOR["int8"]])
    for xi, total, m, d in zip(x, totals, model_us, decode_us):
        label = f"model {m:.2f}\n+ decode {d:.2f}\n= {total:.1f} $\\mu$s" if m > 0 else f"decode only\n= {total:.2f} $\\mu$s"
        ax.text(xi, total * 1.15, label, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"latency per round ($\mu$s, log scale)")
    ax.set_yscale("log")
    ax.set_ylim(top=max(totals) * 6)
    ax.set_title(f"Per-round latency breakdown, $d={distance}$, $p=0.003$")
    fig.tight_layout()
    out_path = out_dir / "fig4_latency_breakdown.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")
    print(
        "    note: FPGA bar uses the real closed-timing kernel's worst-case cycle "
        "count at its actual 100MHz clock; PyMatching decode time is the same CPU "
        "measurement in both non-baseline bars, since decode is not offloaded to "
        "the FPGA in this design."
    )


def fig5_resource_utilization_d5(out_dir: Path) -> None:
    """Device-wide HLS resource utilization at d=5 -- the only distance ever
    synthesized (the kernel is scoped to a fixed distance; see
    docs/LIMITATIONS.md). This is deliberately NOT a "vs distance" sweep:
    resynthesizing at other distances was never done, and this script does
    not estimate or invent numbers for distances that were never built.
    Before/after uses the pre-fix HLS report recovered from git history
    (commit 3c2576f) and the current, post-BRAM-fix report -- both real,
    neither hand-transcribed.
    """
    before = load_json(REPO_ROOT / "results" / "metrics" / "_hls_int8_before_bram_fix.json")
    after = load_json(METRICS / "hls_int8.json")
    if not (before and after):
        print("  skip fig5: missing before/after HLS reports")
        return

    resources = ["BRAM_18K", "DSP", "FF", "LUT"]
    before_pct = [before["utilization_pct"][r] for r in resources]
    after_pct = [after["utilization_pct"][r] for r in resources]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = range(len(resources))
    width = 0.35
    ax.bar([i - width / 2 for i in x], before_pct, width, color=COLOR["int4"], label="before MAC fix (1728-way unroll)")
    ax.bar([i + width / 2 for i in x], after_pct, width, color=COLOR["int8"], label="after MAC fix (this repo)")
    ax.axhline(100.0, color=COLOR["baseline"], linestyle="--", linewidth=1.2, label="100% of device budget")
    ax.set_xticks(list(x))
    ax.set_xticklabels(resources)
    ax.set_ylabel("% of device resources (log scale)")
    ax.set_yscale("log")
    ax.set_title("Device-wide HLS resource utilization at $d=5$ (int8)")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    out_path = out_dir / "fig5_resource_utilization_d5.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")
    print("    note: d=5 only -- the kernel was never resynthesized at other distances.")


def fig6_pareto_precision(sweep: dict, out_dir: Path, distance: int = 5, p: float = 0.003) -> None:
    """Accuracy (LER) vs. quantization cost (nominal bit-width; ternary
    plotted at log2(3) bits, its true information content) at the
    canonical eval point. The Phase 3 finding as a picture: int8 and
    ternary track fp32 closely, int4 is the one real regression.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    baseline_ler = None
    for precision in PRECISIONS:
        rows = [r for r in sweep[precision]["results"] if r["distance"] == distance and r["p"] == p]
        if not rows:
            continue
        r = rows[0]
        baseline_ler = r["baseline_ler"]
        ax.errorbar(
            BIT_WIDTH[precision], r["ler"],
            yerr=[[r["ler"] - r["ler_ci_low"]], [r["ler_ci_high"] - r["ler"]]],
            color=COLOR[precision], marker=MARKER[precision], markersize=9,
            linestyle="none", capsize=3, label=LABEL[precision],
        )
    if baseline_ler is not None:
        ax.axhline(baseline_ler, color=COLOR["baseline"], linestyle="--", linewidth=1.2, label=LABEL["baseline"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nominal weight bit-width (log scale)")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"Accuracy vs. quantization precision\n$d={distance}$, $p={p}$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)
    fig.tight_layout()
    out_path = out_dir / "fig6_pareto_precision.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sweep = load_json(METRICS / "phase6_sweep.json")
    if sweep is None:
        raise SystemExit("results/metrics/phase6_sweep.json not found; run bench/run_phase6_sweep.py first")

    print("Generating figures from committed results/metrics/*.json (no network, no retraining)...")
    fig1_ler_vs_p(sweep, args.out_dir)
    fig2_ler_reduction_factor_vs_d(sweep, args.out_dir)
    fig3_sdr_vs_d(sweep, args.out_dir)
    fig4_latency_breakdown(args.out_dir)
    fig5_resource_utilization_d5(args.out_dir)
    fig6_pareto_precision(sweep, args.out_dir)
    print("done.")


if __name__ == "__main__":
    main()
