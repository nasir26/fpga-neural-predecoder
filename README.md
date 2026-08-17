# fpga-neural-predecoder

FPGA reimplementation, in progress, of the neural pre-decoder concept for
surface-code quantum error correction described by NVIDIA's `Ising-Decoding`
work (see `NOTICE`). This is an independent implementation with no NVIDIA
source code; the FPGA microarchitecture, quantization study, and latency
model are original contributions. Author: Nasir Ali, C-DAC Noida.

**Status: Phase 0 (environment and sanity), Phase 1 (QEC data layer), and
Phase 3 (quantization) done. Phase 2 (pre-decoder model, CPU training)
closed for now with an honest, partially-negative result; R=9 sizing
deferred.** The R=5 pre-decoder trains cleanly and its corrections are
architecturally and numerically sound (verified with an oracle test), but
at the default decision threshold it is not yet a net win over PyMatching
alone, and does not yet demonstrate the distance-generalisation property
the method depends on. Several real bugs were found and fixed along the way
(see `docs/DECISIONS.md` for the full trail); the current numbers are
correct and reproducible, just not yet a positive result for the FP32
baseline. The quantization study (Phase 3) is a clean, positive result in
its own right regardless: INT8 and ternary cost essentially nothing
relative to FP32, only INT4 shows a real regression. See
`claude_code_prompt_ising_fpga.md` in the repository root for the full
phased plan this project follows, and `docs/LIMITATIONS.md` /
`docs/DECISIONS.md` for scope reductions and open issues (notably: X/Z
basis only, no "four orientations"; uniform-depolarizing noise only, the
25-parameter noise model is deferred; Phase 2's model quality gap described
above).

## Phase 2 headline numbers (d=5, p=0.003, R=5, threshold=0.5)

| | LER | vs baseline |
|---|---|---|
| PyMatching only (baseline) | 0.00291 | 1x |
| + pre-decoder (FP32) | 0.01292 | 4.4x worse |

Syndrome density reduction is real (55% at d=5), and a decision threshold
around 0.95 gets within noise of baseline (LER 0.00303) while still removing
26% of syndrome density. LER gets worse, not better, at distances beyond the
R=5 training distance. Full numbers: `results/metrics/predecoder_d*.json`.

## Phase 3 headline numbers: quantization (contribution C1, d=5, p=0.003)

| precision | LER | SDR |
|---|---|---|
| FP32 | 0.01292 | 0.550 |
| INT8 | 0.01258 | 0.552 |
| INT4 | 0.01424 | 0.548 |
| Ternary | 0.01303 | 0.526 |

INT8 and ternary are within noise of FP32; INT4 is the only precision with
a consistent regression (~8-10% relative LER at both d=5 and d=7). QAT
fine-tuning: 3 epochs, 500,000 shots/epoch, from the FP32 checkpoint, one
run per precision. HLS weight headers and binary blobs, with SHA-256, are
exported to `hls/weights_{precision}.h` / `.bin`. Full numbers:
`results/metrics/quantization_sweep.json`.

## Environment

- venv: Python 3.8.10 (see `docs/DECISIONS.md` for why, not 3.11)
- Dependencies: pinned in `requirements.txt`; exact resolved set in
  `docs/env.lock`
- `stim` is pinned to 1.12.1, not the latest release; see
  `docs/DECISIONS.md`
- `torch` is the CPU-only wheel; `torch.cuda.is_available()` is asserted
  False at every environment check

## Setup

```
make venv
make smoke
```

`make smoke` builds a `d=3, rounds=3` rotated surface-code circuit, samples
1000 shots, decodes with PyMatching, and prints the logical error rate, as a
sanity check that the toolchain works end to end.

## Hardware

Target: Xilinx Alveo U55C. This development host happens to have one
installed with a working XRT stack; see `docs/DECISIONS.md` for the full
inventory. Later phases will state plainly whether a given result came from
C simulation, cosimulation, or an actual on-card `hw` run.

## Limitations

Not yet written; will be added starting in Phase 1 and expanded through the
project (`docs/LIMITATIONS.md`).
