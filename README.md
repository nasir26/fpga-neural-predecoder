# fpga-neural-predecoder

FPGA reimplementation of the neural pre-decoder concept for surface-code
quantum error correction described by NVIDIA's `Ising-Decoding` work (see
`NOTICE`). This is an independent implementation with no NVIDIA source code;
the FPGA microarchitecture, quantization study, and latency model are
original contributions. Author: Nasir Ali, C-DAC Noida.

**Method in one paragraph:** a fully convolutional 3D CNN reads a rotated
surface code's syndrome detection volume (built from Stim-simulated circuits)
and predicts, per local edge of the code's detector error model, whether a
local error mechanism fired; those predictions are applied as corrections
(shift-and-XOR against both a fault's detector footprints, plus a fold-in of
each edge's direct effect on the logical observable) before the residual
syndrome is handed to a PyMatching decoder. The network is quantized to
INT8/INT4/ternary via quantization-aware training and exported as fixed-point
weights for a Vitis HLS kernel, synthesized and placed-and-routed to a real
bitstream for a Xilinx Alveo U55C.

**Status: all 8 phases of the governing plan
(`claude_code_prompt_ising_fpga.md`) are done.** The headline result is
mixed, reported honestly rather than reframed: the trained pre-decoder does
not beat a PyMatching-only baseline at its training distance and gets worse,
not better, at larger distances (a negative result); quantization to INT8/
ternary and a real, resource-fitting, timing-closed FPGA bitstream are both
genuine positive results, reported separately from the accuracy claim they
do not rescue. See `paper/paper.pdf` for the full writeup.

## Headline results

| Result | Value | Source |
|---|---|---|
| Baseline LER (PyMatching only, d=5, p=0.003) | 0.00291 | `results/metrics/baseline_d5_r5_z_p0.003.json` |
| Pre-decoder LER (FP32, d=5, p=0.003) | 0.01253 (**4.3x worse** than baseline) | `results/metrics/phase6_sweep.json` |
| Pre-decoder LER reduction factor, d=5 -> d=13 (p=0.003) | 0.23 -> 0.00024 (**~1000x worse**, not better) | `results/metrics/phase6_sweep.json`, Fig. 2 |
| Syndrome density reduction (real, across d=5..13) | 43-56% | `results/metrics/phase6_sweep.json`, Fig. 3 |
| Quantization cost, INT8 / ternary vs FP32 (d=5, p=0.003) | within ~3% relative LER (**negligible**) | `results/metrics/quantization_sweep.json` |
| Quantization cost, INT4 vs FP32 | ~8-10% relative LER regression | `results/metrics/quantization_sweep.json` |
| HLS BRAM_18K, before / after the unroll fix | 1159% of budget -> 2.8% of budget | `docs/DECISIONS.md`, Table in `paper/paper.pdf` |
| Real `hw` bitstream: timing closure | **closed** at 100MHz (WNS +0.003ns, 0 failing endpoints) | `results/metrics/hw_build_int8.json` |
| Real per-shot latency at closed-timing clock | 13.0-15.3 ms (**~4 orders of magnitude** above the ~1us aspirational target) | `results/metrics/hw_build_int8.json` |

Full detail on every row, including the phase-by-phase story of how each
number was reached (and the real bugs found and fixed along the way), is in
the sections below, in `docs/DECISIONS.md`, and in `paper/paper.pdf`.

## Key limitations (see `docs/LIMITATIONS.md` for the full, phase-by-phase log)

- **R=9 was never trained or evaluated.** Every result in this repository is
  for the R=5 (depth-2, receptive-field-5) pre-decoder only; R=9 training was
  scoped out early as a ~20-hour CPU-only commitment and deliberately
  deferred, not attempted and hidden.
- **The pre-decoder does not beat the baseline it is meant to improve on**,
  and its disadvantage grows, not shrinks, past its training distance --
  the opposite of the distance-generalization property the fully
  convolutional architecture is intended to provide. See `paper/paper.pdf`
  Section IX for the candidate explanations (undertrained relative to the
  reference recipe's compute budget; a training target that is, by
  construction, a strict subset of the true error process).
- **The HLS kernel is scoped to a single fixed code distance (d=5)**, not
  runtime-general, because the observable-parity lookup table it uses was
  not verified to be a fixed geometric rule independent of distance.
- **No result in this repository comes from running on real FPGA hardware.**
  A real, timing-closed bitstream exists (`hls/build_hw/predecoder_kernel.xclbin`,
  gitignored, rebuildable from source), but `host/run_hw.py`'s real-hardware
  code path has never been exercised against the card; every reported
  latency/accuracy number from the host pipeline comes from its labeled,
  honest software fallback (`"simulated": true` in every such result).
- **Real per-shot latency (13.0-15.3 ms) remains roughly four orders of
  magnitude above the method's own ~1us aspirational target.** This is not
  primarily a clock-speed artifact: the dominant cost is a direct, load-
  bearing consequence of the same fix that resolved the BRAM overshoot
  (trading unrolled parallelism for on-chip memory budget).
- Noise model is Stim's uniform-depolarizing channel only; the full
  25-parameter circuit-level noise model is not implemented. X/Z memory
  basis only, not "four orientations" (the rotated surface code's boundary
  symmetry makes this a relabeling, not new physics).

## Hardware requirements

- **Target device:** Xilinx Alveo U55C (`xcu55c-fsvh2892-2L-e`,
  `xilinx_u55c_gen3x16_xdma_3_202210_1` platform). All HLS/synthesis/
  place-and-route results in this repository were built against this exact
  platform file.
- **To rebuild the bitstream:** Vitis/Vitis HLS 2023.2 and Vivado 2023.2 (the
  `v++`/`vitis_hls` toolchain), plus the platform `.xpfm` installed locally.
  No GPU is used or required anywhere in this project.
- **To run inference/training/benchmarks:** any x86_64 Linux host with the
  pinned CPU-only Python environment below; no FPGA card is required for
  anything except an actual on-card `hw` run, which this project has not
  performed (see Limitations above).
- **To load the built bitstream onto a real card:** XRT (this project used
  2.15.225) and a physical Alveo U55C; not required to reproduce any other
  number in this repository.

## Reproduce

```bash
# Environment (CPU-only; no GPU, no Docker)
make venv
make smoke      # sanity check: builds a d=3 circuit, decodes 1000 shots
make test       # 76 unit/regression tests

# Phase 1: baseline (PyMatching only)
make baseline

# Phase 2: train the FP32 pre-decoder (single CPU, ~1.7 hours at defaults)
python src/model/train.py
python bench/run_predecoder.py --checkpoint models/predecoder_R5.pt \
    --distances 5 7 9 11 --p 0.003 --shots 100000 --timing-shots 1000

# Phase 3: quantization-aware fine-tuning and sweep
python src/model/train_qat.py --precision int8
python src/model/train_qat.py --precision int4
python src/model/train_qat.py --precision ternary
python bench/sweep_precision.py --fp32-checkpoint models/predecoder_R5.pt \
    --distances 5 7 --p 0.003 --shots 100000 --timing-shots 1000

# Phase 4: HLS csim/csynth/cosim (Vitis HLS 2023.2; no hardware needed)
source /opt/xilinx/xrt/setup.sh && source /tools/Xilinx/Vitis/2023.2/settings64.sh
cd hls && vitis_hls -f run_hls.tcl && cd ..

# Phase 4/5 (optional, hours, needs the real platform file):
# a real hw bitstream build
cd hls
v++ -c -t hw --platform <path-to>/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
    -k predecoder_kernel -I. -o build_hw/predecoder_kernel.xo predecoder_kernel.cpp
v++ -l -t hw --platform <path-to>/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm \
    --config link_100mhz.cfg -o build_hw/predecoder_kernel.xclbin build_hw/predecoder_kernel.xo
cd ..

# Phase 5: host integration (falls back to the honest software path if no
# xclbin/card is available, which is the normal case)
python host/run_hw.py --shots 10000

# Phase 6: full benchmark grid (hours) and figures (seconds, from committed data)
python bench/run_phase6_sweep.py
make figures

# Phase 7: paper (needs a LaTeX toolchain; see paper/build.sh for a no-root install path)
make paper
```

Every command above writes to `results/metrics/*.json` or `results/figures/`;
none of this project's reported numbers were computed any other way.

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

| | original | after BRAM fix (2026-08-19) |
|---|---|---|
| Estimated Fmax | 136.99 MHz | 360.23 MHz |
| Latency (worst case) | 48,077 cycles / 0.481 ms @ 100 MHz | 1,528,325 cycles / 5.094 ms @ ~300 MHz (v++ default clock, not the same target -- see `docs/DECISIONS.md`) |
| DSP / FF / LUT | 13.5% / 9.8% / 32.9% | 3.1% / 11.1% / 28.1% |
| **BRAM_18K** | **46,716 / 4,032 -- 1159%, over 11x budget** | **114 / 4,032 -- 2.8%, under budget** |

This is a first-pass, correctness-prioritized synthesis (two pragma choices
caused multi-hour synthesis runaways before being diagnosed and fixed; see
`docs/DECISIONS.md`), not a resource-optimal one. The original BRAM
overshoot (11x over budget) is now fixed: the dominant driver was a
1728-way fully unrolled MAC loop (`dz,dy,dx,ic` all unrolled for the wide
hidden layer), not the frame buffer. De-unrolling the `ic` dimension to a
27-way spatial unroll plus a serially-pipelined channel reduction cut
BRAM_18K to 2.8% of budget, verified bit-exact via csim before
resynthesizing -- at the cost of ~32x worse latency on top of an already
far-over-target number (per-shot latency remains many orders of magnitude
above the paper's own ~1us aspirational target; no shot-level pipelining
yet). The kernel is scoped to a single fixed distance (d=5), not
runtime-general; see `docs/LIMITATIONS.md`.

**Update, 2026-08-19: a real, timing-closed `.xclbin` was built.**
`v++ -l -t hw` (full place-and-route against the U55C platform shell), in
two attempts: the first, at v++'s own default ~300 MHz target (3h54m),
produced a well-formed but timing-unclosed bitstream (WNS -0.609 ns).
Rebuilding just the link step at an explicit 100 MHz target
(`hls/link_100mhz.cfg`, 2h33m, same `.xo` reused) closed it: **WNS
+0.003 ns, TNS 0.000 ns, 0 of 700,988 endpoints failing**. Verified
well-formed via `xclbinutilinfo` (correct kernel name, platform, HBM port
mapping). Kernel-scoped post-route utilization (excludes the platform
shell): LUT 8.6% / FF 3.4% / BRAM 4.3% / DSP 3.7% of the kernel's user
budget. Real latency at the actual 100 MHz clock: 13.0-15.3 ms/shot,
still roughly 15,000x the paper's ~1us aspirational target. This is the
first result in the project that is simultaneously within budget and
timing-closed; it has not yet been run on the actual card (`run_hw.py`'s
real-hardware path is still an unexercised stub). See
`docs/DECISIONS.md` and `docs/LIMITATIONS.md` for the full trail and
`results/metrics/hw_build_int8.json` for the numbers.

## Phase 5: host integration

`host/host_predecoder.cpp` (XRT C++: HBM bank assignment via
`kernel.group_id()`, a ping-pong/ring buffer of 4 slots for overlap across
shots, per-stage timing) and `host/run_hw.py` (the same design via pyxrt)
are written against the real XRT API -- the C++ host compiles and links
against the real `libxrt_coreutil.so` on this host -- but **have not run
against real hardware**: a real xclbin now exists (see Phase 4 update
above), but loading it onto the card and running `run_on_hardware` has not
been attempted. `run_hw.py` falls back to the golden model whenever no
xclbin is available (the current, actual state), and labels every such
result `"simulated": true`, loudly, never silently:

```
$ python host/run_hw.py --shots 10000
WARNING: no usable xclbin ... Falling back to the golden model.
RESULTS ARE SIMULATED, NOT FROM REAL HARDWARE.
wrote results/metrics/host_run_d5_p0.003.json (simulated=True)
```

Full numbers: `results/metrics/host_run_d5_p0.003.json`.

## Phase 6: benchmarks and figures (R=5 only; R=9 remains deferred)

`bench/run_phase6_sweep.py` evaluates all four trained R=5 checkpoints
(fp32/int8/int4/ternary) across the full distance x noise-rate grid
(d in {5,7,9,11,13}, p in {0.001,0.002,0.003,0.005}, 80 combinations,
100k shots each), generating any missing PyMatching-only baselines along
the way. `make figures` (`scripts/make_figures.py`) regenerates every
figure below from the committed `results/metrics/*.json` only -- no
network access, no retraining, no re-running HLS/v++.

| | |
|---|---|
| ![LER vs p](results/figures/fig1_ler_vs_p.png) | ![LER reduction factor vs d](results/figures/fig2_ler_reduction_factor_vs_d.png) |
| ![SDR vs d](results/figures/fig3_sdr_vs_d.png) | ![Latency breakdown](results/figures/fig4_latency_breakdown.png) |
| ![Resource utilization](results/figures/fig5_resource_utilization_d5.png) | ![Precision Pareto front](results/figures/fig6_pareto_precision.png) |

The honest finding, stated visually rather than just in prose: every
precision's LER curve sits *above* PyMatching-only across the tested
noise range (Fig 1), and the pre-decoder's relative disadvantage widens
by roughly three orders of magnitude from d=5 to d=13 (Fig 2) -- this
grid confirms, at full scale, the distance-generalisation gap already
found in Phase 2. Syndrome density reduction stays real and substantial
regardless (Fig 3), a separate and still-true claim. Fig 5 is
deliberately not a resource-vs-distance sweep: the kernel was only ever
synthesized at d=5. Full numbers: `results/metrics/phase6_sweep.json`.

## Phase 7: paper

`paper/paper.pdf` (IEEE format, `IEEEtran`, built from `paper/paper.tex`
via `make paper`) reports the full project honestly: the pre-decoder's
negative accuracy result as the headline finding, the quantization
study as a genuine positive result, and the real, timing-closed FPGA
build as a genuine positive result, without conflating the three. Every
number in it is sourced from a committed `results/metrics/*.json` file.
See `paper/build.sh` for the LaTeX toolchain needed to rebuild it
(a user-space TinyTeX install, no root required).

## Environment

- venv: Python 3.8.10 (see `docs/DECISIONS.md` for why, not 3.11)
- Dependencies: pinned in `requirements.txt`; exact resolved set in
  `docs/env.lock`
- `stim` is pinned to 1.12.1, not the latest release; see
  `docs/DECISIONS.md`
- `torch` is the CPU-only wheel; `torch.cuda.is_available()` is asserted
  False at every environment check
