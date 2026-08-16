# Claude Code Prompt: FPGA Neural Pre-Decoder (Ising-Decoding reimplementation)

Paste everything below the line into Claude Code (VS Code extension) as the opening
message of a new session in an empty repository directory.

---

## ROLE

You are implementing a research-grade FPGA reimplementation of the neural
pre-decoder concept from NVIDIA's `Ising-Decoding` repository
(https://github.com/NVIDIA/Ising-Decoding), targeting a Xilinx Alveo U55C
accelerator card. The deliverable is a public GitHub repository plus a
publication-quality paper. Author on all artifacts: **Nasir Ali, C-DAC Noida**.

You are not porting NVIDIA's code. You are building an independent
implementation of the published method, with FPGA-specific contributions that
the reference implementation does not have.

## HARD CONSTRAINTS (violating any of these fails the task)

1. **No model weights may be downloaded.** Do not run `hf download`,
   `huggingface-cli`, `hf auth login`, or fetch any `.pt` / `.safetensors` /
   `.onnx` checkpoint. The NVIDIA Hugging Face checkpoints are access-gated and
   out of scope. All neural weights in this project are trained locally from
   scratch on CPU.
2. **No LLM runs locally.** Do not run `ollama pull`, do not install a local
   model runtime, do not install LM Studio / GPT4All / any desktop app. If any
   pipeline step needs an LLM (for example, docstring generation or log
   summarisation), it must call Ollama Cloud over HTTPS:
   `OLLAMA_HOST=https://ollama.com` with the key read from the environment
   variable `OLLAMA_API_KEY`. Never hardcode a key. Never write a key to a file
   that is committed.
3. **No Docker, no containers, no system-wide installs.** Everything runs
   inside a project-local Python virtual environment created in the workspace.
   All work happens through the VS Code integrated terminal and the workspace
   filesystem.
4. **No CUDA / GPU code paths.** No `cupy`, no `cuquantum-python`, no
   `cudaq-qec`, no `tensorrt`, no `nvidia-modelopt`. PyTorch must be the CPU
   wheel only. If a script cannot run without a GPU, it does not belong in this
   repository.
5. **Allowed Python dependencies only:** `stim`, `pymatching`, `numpy`,
   `scipy`, `torch` (CPU wheel), `matplotlib`, `pandas`, `pytest`, `pyyaml`,
   `tqdm`, `pyxrt` (host side, if XRT is present). Ask before adding anything
   else. Pin every version in `requirements.txt`.
6. **Before any `pip install`, print the resolved wheel list and total download
   size, then stop and ask for confirmation.**

## HARDWARE / TOOLCHAIN CONTEXT

- Target: Xilinx Alveo U55C, HBM2 (32 GB, 32 pseudo-channels), Vitis HLS C++.
  Up to 4 cards available, but the primary result must be single-card.
- Existing local experience to build on and cross-reference (do not copy code
  blindly, re-derive): HLS C++ kernels over 512-bit AXI to HBM, LUT-based
  distance-3 decoders, Union-Find and MWPM decoder work, ring-buffer flow
  control, and a 4-stage pipelined syndrome to difference to decode to
  correction datapath.
- Assume Vitis/Vitis HLS is installed but may not be licensed for a full
  bitstream build in every session. **Every stage must be runnable and
  meaningful in C simulation and C/RTL cosimulation alone.** Hardware
  emulation and `hw` build are optional final stages gated behind a flag.
- Development machine is CPU-only. All training, dataset generation, and
  baseline decoding run on CPU.

## THE METHOD YOU ARE IMPLEMENTING

Restated from the NVIDIA papers (Chamberland, Olle, Li, Thornton, Baratta,
"Fast and accurate AI-based pre-decoders for surface codes", arXiv:2604.12841,
2026; and the companion color-code paper):

1. A rotated surface-code memory experiment produces a detector syndrome volume
   across space and time.
2. A 3D convolutional neural network consumes that volume and predicts local
   corrections that remove a large fraction of the detection events.
3. The residual syndrome is passed to a standard global decoder (PyMatching for
   surface codes), which produces the final logical decision.
4. Because the pre-decoder is fully convolutional, a model trained at
   `distance = n_rounds = R` (its receptive field) can be evaluated at larger
   distances and round counts without retraining.

Key reported metrics you must reproduce in spirit: logical error rate (LER),
LER reduction factor, syndrome density reduction (SDR), and PyMatching speedup
from the reduced syndrome density.

**Scope for this project: surface code only. Color code and Chromobius are
explicitly out of scope.**

## WHAT MAKES THIS PROJECT NOVEL (the paper's contribution)

The reference implementation runs the pre-decoder at FP8 on a datacenter GPU.
This project asks a different question:

> Can the same pre-decoder run as a fixed-point, streaming, fully pipelined 3D
> convolution on an HBM-attached FPGA, at a per-round latency compatible with
> the ~1 microsecond syndrome cycle of a superconducting processor, and does the
> quantization cost any logical accuracy?

Contributions to establish and defend with data:
- C1: A quantization study (FP32 -> INT8 -> INT4 -> ternary weights) measuring
  LER, SDR, and PyMatching speedup degradation at each precision.
- C2: A line-buffer / sliding-window 3D convolution microarchitecture in HLS
  C++ with initiation interval 1 per syndrome voxel, sized so the entire weight
  set is resident in on-chip BRAM/URAM with zero DRAM weight traffic.
- C3: A latency and resource model as a function of code distance `d`,
  receptive field `R`, and channel width, validated against cosimulation.
- C4: An end-to-end latency budget comparing FPGA pre-decode plus host
  PyMatching against host-only PyMatching, with the syndrome transfer cost
  counted honestly.
- C5: An open, reproducible artifact: fixed seeds, committed Stim circuits,
  committed metrics, one command to regenerate every figure.

## REPOSITORY LAYOUT (create exactly this)

```
fpga-neural-predecoder/
  README.md
  LICENSE                     # Apache-2.0
  NOTICE                      # attribution to NVIDIA Ising-Decoding + papers
  requirements.txt
  .gitignore
  Makefile                    # one entry point per stage
  conf/
    default.yaml              # single source of truth for all parameters
  src/
    qec/
      circuits.py             # Stim rotated surface-code memory circuits
      dem.py                  # detector error model, check matrices
      sampling.py             # detector sampling, diff frames, seeds
      baseline.py             # PyMatching-only baseline decoder + timing
      metrics.py              # LER (with Wilson CI), SDR, speedup, latency
    model/
      predecoder.py           # 3D CNN, configurable R and channel widths
      quantize.py             # symmetric per-channel INT8/INT4/ternary QAT
      train.py                # CPU training loop
      export.py               # weights -> C header + .bin for HLS
      golden.py               # bit-exact NumPy replica of the HLS datapath
  hls/
    predecoder_kernel.cpp     # HLS C++ streaming 3D conv kernel
    predecoder_kernel.h
    line_buffer.h             # sliding-window buffers
    weights.h                 # generated, do not edit by hand
    tb_predecoder.cpp         # C testbench, reads vectors from tests/vectors
    run_hls.tcl               # csim, csynth, cosim; hw build behind a flag
  host/
    host_predecoder.cpp       # XRT host, 512-bit AXI to HBM, ring buffer
    run_hw.py                 # pyxrt driver, falls back to golden if no card
  bench/
    run_baseline.py
    run_predecoder.py
    sweep_distance.py
    sweep_precision.py
    make_figures.py
  tests/
    test_circuits.py
    test_golden_vs_torch.py   # bit-exactness, tolerance zero
    test_metrics.py
    vectors/                  # committed small test vectors
  results/
    metrics/*.json            # committed, machine-generated
    figures/*.pdf
  paper/
    main.tex                  # IEEE conference format
    refs.bib
    figures/                  # symlink or copy of results/figures
  docs/
    ARCHITECTURE.md
    REPRODUCE.md
    LIMITATIONS.md
```

## PHASED EXECUTION PLAN

Work through these phases in order. **At the end of each phase, stop, print a
short status report, and wait for my go-ahead before starting the next.** Do
not run ahead.

### Phase 0: environment and sanity

- Create the venv, install pinned deps (after the size confirmation described
  above), record `pip freeze` into `docs/env.lock`.
- Assert no CUDA: fail loudly if `torch.cuda.is_available()` is True or if any
  forbidden package is importable.
- Print Vitis HLS version if present; if absent, record that and continue in
  simulation-only mode.
- Smoke test: build a `d=3, rounds=3` Stim circuit, sample 1000 shots, decode
  with PyMatching, print LER.

### Phase 1: QEC data layer

- Rotated surface-code memory circuits in both X and Z basis, all four
  orientations, circuit-level depolarizing noise parameterised by a single `p`.
  Support a 25-parameter circuit-level noise model (SPAM, idle, CNOT Pauli
  channels) as a dict, with the uniform-depolarizing mapping as the default.
- Detector sampling with an explicit seed. Emit the decoding volume as a
  `(shots, T, H, W)` uint8 tensor plus observable flips.
- Diff frames: XOR of consecutive rounds, which is the actual network input.
- Baseline: PyMatching decode, single-shot timing (`batch_size=1`) in
  microseconds per round, LER with a Wilson confidence interval.
- Write `results/metrics/baseline_*.json`.
- Tests: detector counts match `stim`'s own report; LER at `d=3,5,7`,
  `p=0.003` sits on the expected threshold curve within CI.

### Phase 2: pre-decoder model, CPU training

- 3D CNN, fully convolutional, no pooling, `same` padding, kernel 3, receptive
  field `R` set by depth. Start at `R=5` (depth 2) and `R=9` (depth 4).
  Channel widths configurable, default `[64, 64, 64, out]`.
- Output head predicts per-voxel correction proposals; loss is
  `BCEWithLogitsLoss` against the ground-truth fault frame derived from the DEM.
- Training runs at `distance = n_rounds = R` per the method. Because this is
  CPU-only, use a reduced shot budget: default 2e6 shots per epoch, 20 epochs,
  batch 256. **Log this as a deliberate deviation from the reference recipe
  (67M shots per epoch, 100 epochs, 8 GPUs) and carry it into
  `docs/LIMITATIONS.md` and the paper's limitations section.**
- Noise upscaling during training: rescale the noise vector so the largest
  effective fault-channel probability reaches 6e-3, evaluate on the original
  noise. Make this switchable.
- Checkpoint to `models/` (gitignored, weights are regenerated not committed);
  commit only the exported quantized weight headers and the metrics.
- Evaluate at `d = 5, 7, 9, 11` with the `R=9` model to demonstrate the
  distance-generalisation property. Report LER reduction factor and SDR.

### Phase 3: quantization and golden model

- Symmetric per-output-channel quantization: INT8, INT4, ternary weights;
  activations as `ap_fixed`-compatible fixed point. Quantization-aware
  fine-tuning for a few epochs at each precision.
- `golden.py`: a NumPy implementation that is **bit-exact** with the intended
  HLS integer datapath, including accumulator width, rounding mode, and
  saturation. `test_golden_vs_torch.py` must pass with zero tolerance against
  the quantized PyTorch model.
- Export weights to `hls/weights.h` (constexpr arrays) plus a `.bin` for the
  host path. Include a SHA-256 of the weight blob in the header comment and in
  the metrics JSON.
- Sweep and record: LER, LER reduction factor, SDR, PyMatching speedup at every
  precision. This is contribution C1.

### Phase 4: HLS kernel

- `predecoder_kernel.cpp`: `hls::stream` dataflow. Syndrome voxels stream in
  from a 512-bit AXI master mapped to one HBM pseudo-channel; corrections
  stream out to a second.
- Line buffers hold `2 * W + kernel` rows per layer so each 3x3x3 window is
  formed once; target `II=1` on the innermost pixel loop.
- Weights fully partitioned into BRAM/URAM, zero weight traffic at runtime.
- Fixed-point types via `ap_fixed` / `ap_int` matching the golden model
  exactly.
- Pragmas: `HLS DATAFLOW` at the top, `PIPELINE II=1` on pixel loops,
  `ARRAY_PARTITION` on window and weight arrays, `INTERFACE m_axi` with
  explicit `max_read_burst_length` and `num_write_outstanding`.
- `tb_predecoder.cpp` reads committed vectors from `tests/vectors/` and asserts
  bit-equality with golden output.
- `run_hls.tcl`: csim, then csynth (record latency, II, LUT/FF/BRAM/URAM/DSP),
  then cosim. Guard the `hw` build behind `BUILD_HW=1`; do not launch a
  multi-hour build without being asked.
- Record the synthesis report numbers into `results/metrics/hls_*.json`
  automatically. Never hand-transcribe a resource number into the paper.

### Phase 5: host integration

- `host_predecoder.cpp` with XRT: HBM bank assignment, ping-pong buffers, ring
  buffer flow control, event-based timing.
- `run_hw.py` via `pyxrt`. If no card is present, it must fall back to the
  golden model and clearly label the results as simulated, never silently.
- End-to-end latency accounting: host-to-device transfer, kernel, device-to-host,
  residual PyMatching decode. Report microseconds per round and the honest total,
  not the kernel time alone.

### Phase 6: benchmarks and figures

- Sweeps: `d` in {5,7,9,11,13}, `p` in {0.001, 0.002, 0.003, 0.005}, precision
  in {fp32, int8, int4, ternary}, `R` in {5, 9}.
- Shot counts sized so LER confidence intervals are visible on the plots; state
  the count in every caption.
- Figures: LER vs p (threshold), LER reduction factor vs d, SDR vs d, latency
  breakdown stacked bar, resource utilisation vs d, accuracy vs precision
  Pareto front.
- `make figures` regenerates all of them from committed JSON with no network
  access and no re-training.

### Phase 7: paper

`paper/main.tex`, IEEE conference format, 6 to 8 pages. Structure:

1. Abstract, quantifying the headline latency and the LER cost of quantization.
2. Introduction: real-time decoding latency budget for superconducting QEC, why
   pre-decoding helps, why FPGAs matter for the sub-microsecond regime.
3. Background: rotated surface code, detector error models, MWPM and PyMatching,
   the neural pre-decoder concept as published by NVIDIA (cite both papers).
4. Method: network architecture, quantization scheme, HLS microarchitecture with
   a dataflow figure, resource and latency model.
5. Experimental setup: Stim circuits, noise model, shot counts, seeds, toolchain
   versions, hardware.
6. Results: the six figure families above.
7. Discussion: where the FPGA wins, where the GPU wins, what the crossover
   distance is.
8. Limitations: reduced training budget versus the reference recipe, single
   code family, simulation-only stages if the bitstream did not build, no
   comparison against the gated pretrained checkpoints.
9. Reproducibility statement, then references.

Writing rules for the paper and all documentation:
- **No em dashes.** Use commas, semicolons, colons, or parentheses.
- Every number in the text must trace to a file in `results/metrics/`. If a
  number is not in a JSON file, it does not go in the paper.
- Do not claim to beat NVIDIA's published numbers. You are on different hardware
  with a smaller training budget. Frame results as a hardware-mapping study.
- Do not fabricate a bitstream result. If only cosimulation ran, the paper says
  cosimulation.

### Phase 8: repository hygiene and publication

- SPDX header on every source file:
  `// SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida`
  `// SPDX-License-Identifier: Apache-2.0`
- `NOTICE` credits NVIDIA's Ising-Decoding repository (Apache-2.0) and both
  papers as the source of the method, and states clearly that this is an
  independent implementation containing no NVIDIA source code.
- `README.md`: what this is, one-paragraph method summary, headline results
  table, exact reproduce commands, hardware requirements, honest limitations
  near the top.
- GitHub Actions CI, CPU only: lint, `pytest`, golden-vs-torch bit-exactness, a
  60-second `d=5` smoke train and eval. No HLS in CI.
- Commit in logical increments with meaningful messages, one commit per
  completed sub-task, not one giant commit. Do not push until I confirm the
  remote and the repository name.

## BEHAVIOURAL RULES FOR YOU (Claude Code)

- Stop at every phase boundary and report. Do not chain phases.
- Before writing more than 100 lines of new code, show me the interface
  (function signatures, tensor shapes, dtypes) and wait.
- Never invent a benchmark number, a resource utilisation figure, or an LER. If
  a run did not happen, say it did not happen.
- If a design decision has a real tradeoff (channel width versus BRAM, INT4
  versus INT8, `R=5` versus `R=9`), present both sides and ask rather than
  silently choosing.
- If any instruction here conflicts with what the hardware or toolchain
  actually supports, stop and tell me instead of working around it quietly.
- Keep a running `docs/DECISIONS.md` log: date, decision, alternatives
  considered, reason.
