# fpga-neural-predecoder

FPGA reimplementation, in progress, of the neural pre-decoder concept for
surface-code quantum error correction described by NVIDIA's `Ising-Decoding`
work (see `NOTICE`). This is an independent implementation with no NVIDIA
source code; the FPGA microarchitecture, quantization study, and latency
model are original contributions. Author: Nasir Ali, C-DAC Noida.

**Status: Phase 0 (environment and sanity), Phase 1 (QEC data layer),
Phase 3 (quantization), Phase 4 (HLS kernel: csim/csynth/cosim), and Phase 5
(host integration, code written and API-verified but not run against real
hardware) done. Phase 2 (pre-decoder model, CPU training) closed for now
with an honest, partially-negative result; R=9 sizing deferred.** The R=5
pre-decoder trains cleanly and its corrections are
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

## Phase 4 headline numbers: HLS kernel (INT8, d=5 only, `xcu55c-fsvh2892-2L-e`)

`hls/predecoder_kernel.cpp`: line-buffered 3D conv (not a full-volume
buffer) x2, 1x1x1 output conv, edge-correction + observable-parity stage,
all four DATAFLOW-connected. csim, csynth, and cosim (real RTL simulation,
not just the C model) all pass, cosim bit-exact 10/10 against
`src/model/golden.py`. Real numbers, auto-parsed into
`results/metrics/hls_int8.json` (`scripts/record_hls_metrics.py`), reported
honestly including the parts that are not yet good:

| | value |
|---|---|
| Estimated Fmax | 136.99 MHz |
| Latency (worst case) | 48,077 cycles / 0.481 ms at 100 MHz target |
| DSP / FF / LUT | 13.5% / 9.8% / 32.9% of device |
| **BRAM_18K** | **46,716 / 4,032 available -- 1159%, over 11x budget** |

This is a first-pass, correctness-prioritized synthesis (two pragma choices
caused multi-hour synthesis runaways before being diagnosed and fixed; see
`docs/DECISIONS.md`), not a resource-optimal one: it would not fit on the
real device as-is, and per-shot latency is ~350-500x the paper's own ~1us
aspirational target (no shot-level pipelining yet). The kernel is scoped to
a single fixed distance (d=5), not runtime-general; see
`docs/LIMITATIONS.md`. The `hw` bitstream build has not been run.

## Phase 5: host integration

`host/host_predecoder.cpp` (XRT C++: HBM bank assignment via
`kernel.group_id()`, a ping-pong/ring buffer of 4 slots for overlap across
shots, per-stage timing) and `host/run_hw.py` (the same design via pyxrt)
are written against the real XRT API -- the C++ host compiles and links
against the real `libxrt_coreutil.so` on this host -- but **have not run
against real hardware**: no xclbin exists (the `hw` build was not
attempted; see Phase 4). `run_hw.py` falls back to the golden model
whenever no xclbin is available (the current, actual state), and labels
every such result `"simulated": true`, loudly, never silently:

```
$ python host/run_hw.py --shots 10000
WARNING: no usable xclbin ... Falling back to the golden model.
RESULTS ARE SIMULATED, NOT FROM REAL HARDWARE.
wrote results/metrics/host_run_d5_p0.003.json (simulated=True)
```

Full numbers: `results/metrics/host_run_d5_p0.003.json`.

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
