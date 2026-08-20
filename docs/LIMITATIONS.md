# Limitations

This file accumulates deliberate scope reductions and known gaps, each with
the reason it was made. Entries are added as each phase introduces them; see
`docs/DECISIONS.md` for the full reasoning trail behind each one.

## Phase 1: QEC data layer

- **Two orientations, not four.** `stim.Circuit.generated` exposes exactly
  `rotated_memory_x` and `rotated_memory_z` for the rotated surface code.
  There is no built-in third or fourth orientation, and the rotated surface
  code's boundary symmetry (smooth/rough boundary duality) makes a
  hand-built third or fourth orientation physically equivalent to X/Z memory
  under relabeling, so nothing physically new would be measured by building
  them. Only X-basis and Z-basis memory experiments are implemented. Decided
  with the user on 2026-08-16; see `docs/DECISIONS.md`.

- **Uniform-depolarizing noise only; the 25-parameter circuit-level noise
  model is not implemented.** `stim.Circuit.generated` accepts four scalar
  noise knobs (after-Clifford depolarization, before-round data
  depolarization, before-measure flip, after-reset flip), all driven by a
  single `p` in this project. A true 25-parameter model (independent SPAM,
  idle, and all 15 two-qubit Pauli-channel probabilities) requires
  hand-constructing the circuit with explicit `PAULI_CHANNEL_1`/
  `PAULI_CHANNEL_2` injection instead of using Stim's generator, which is a
  much larger, higher-risk undertaking: a subtle circuit-construction bug
  there would silently corrupt every downstream LER number. `NoiseModel` in
  `src/qec/circuits.py` already has all 25 field names defined and raises
  `NotImplementedError` for `kind="full25"`, so the interface is
  future-proofed but the implementation is deferred. Decided with the user
  on 2026-08-16.

## Phase 2: pre-decoder model, CPU training

- **Reduced training budget versus the reference recipe.** The reference
  method (NVIDIA) trains with roughly 67M shots/epoch, 100 epochs, across 8
  GPUs. This project trains on a single CPU with 2,000,000 shots/epoch and
  20 epochs by default (`conf/default.yaml: training`), matching the plan's
  own stated deliberate reduction. Measured on this host (64 threads, no
  GPU): one training step is ~37ms/batch at R=5 (depth 2) but ~438ms/batch
  at R=9 (depth 4), so even this reduced recipe is roughly 1.7 hours total
  at R=5 and roughly 20 hours at R=9. R=5 was trained first to validate the
  pipeline before committing to the much longer R=9 run; see
  `docs/DECISIONS.md`.
- **Ground-truth target covers a subset of detection events by
  construction.** The training target is the XOR-parity of DEM error
  mechanisms that fired and touch at most 2 detectors ("local" errors).
  Errors touching 3+ detectors are deliberately excluded from the target and
  left for PyMatching; this matches the method's framing of removing "a
  large fraction," not all, of detection events, but means the pre-decoder
  is architecturally incapable of ever fully resolving those nonlocal
  events on its own. See `docs/DECISIONS.md`.
- **The R=5 model does not yet demonstrate a net LER improvement at the
  default decision threshold, and does not yet demonstrate distance
  generalisation.** After fixing three real bugs found during evaluation
  (per-voxel parity breaking; local corrections not tracking their own
  logical-observable effect; an XOR-parity bug in that same observable
  tracking), LER at d=5, p=0.003, threshold=0.5 is 0.0129, still about 4.4x
  worse than the PyMatching-only baseline (0.00291), though a threshold
  around 0.95 gets within noise of baseline (0.00303) while still removing
  26% of syndrome density. LER gets worse, not better, at distances beyond
  the R=5 training distance (d=7, 9, 11), the opposite of the fully
  convolutional distance-generalisation property the method is supposed to
  have. Full numbers and the diagnosis trail are in `docs/DECISIONS.md`
  and `results/metrics/predecoder_d{5,7,9,11}_R5_p0.003.json`.
- **The evaluation script's per-shot latency numbers required a fix to be
  trustworthy.** `bench/run_predecoder.py`'s timing loop was letting
  PyTorch use all 64 threads for every tiny single-shot (batch=1) forward
  call, which is unrealistic for a real single-shot deployment and made the
  measurement highly sensitive to any concurrent load on the machine (an
  accidental duplicate background run pushed load to ~92-99 on this 64-core
  host and inflated the reading to >100ms/shot; the same evaluation in
  isolation gave ~1ms/shot). Fixed by pinning the timing loop to
  `torch.set_num_threads(1)`. LER, SDR, and ler_reduction_factor were never
  affected by this; only mean_latency_us_per_shot/round. See
  `docs/DECISIONS.md`.

## Phase 3: quantization and golden model

- **The quantization study (contribution C1) is conducted on an FP32
  baseline that does not itself beat PyMatching alone.** This was a
  deliberate, explicit choice, confirmed with the user before starting
  Phase 3: the quantization question (does INT8/INT4/ternary cost accuracy
  relative to FP32) is answered cleanly and honestly regardless of whether
  that FP32 baseline is itself a net win over the Phase 1 baseline. It is
  not yet a net win (see the Phase 2 entries above). The quantization
  sweep's own finding is real and useful on its own terms: INT8 and ternary
  are within noise of FP32, INT4 shows a consistent ~8-10% relative LER
  regression. See `docs/DECISIONS.md`, 2026-08-17.
- **"Zero tolerance" for golden.py vs the QAT PyTorch model is defined on
  the correction decision (exact), not literal bit-equality of the
  underlying float values (1e-6 tolerance instead).** Floating-point
  addition is not associative, so PyTorch's BLAS-based `F.conv3d` and
  golden.py's tensordot-based integer accumulation cannot be made literally
  bit-identical without abandoning `F.conv3d` (and the batched-training
  speed it provides) entirely. Decided with the user; see
  `docs/DECISIONS.md`, 2026-08-17.
- **Bias must be quantized to an integer at the accumulator's implicit
  scale**, not kept at full float precision as first implemented: a real
  integer accumulator cannot add an arbitrary-precision float bias. Fixed
  before this caused any incorrect result to be reported; see
  `docs/DECISIONS.md`.

## Phase 4: HLS kernel

- **The kernel is scoped to a single fixed distance, d=5**, not runtime-
  general across distances, because the observable-parity lookup table is
  not (yet, verifiably) a fixed geometric rule independent of distance; see
  `docs/DECISIONS.md`, 2026-08-17. The CNN datapath itself is architecturally
  distance-general.
- **This is a first-pass, correctness-prioritized synthesis, not a
  resource-optimal one.** Two pragma choices caused csynth to run for hours
  without converging (fully-unrolled 110,592-way MAC parallelism; a fully
  array-partitioned, data-dependently-indexed correction buffer forced to
  II=1) before being diagnosed and fixed; see `docs/DECISIONS.md` for the
  full story. With both fixed, csynth completes in about an hour and all
  loop constraints are satisfied, but the resulting design originally used
  **1159% of the U55C's BRAM_18K budget** (46,716 used vs 4,032 available)
  and had per-shot latency of 0.35-0.48 ms, roughly 350-500x the paper's own
  ~1us aspirational target (no shot-level pipelining is implemented; each
  shot runs mostly-sequentially through the 4 DATAFLOW stages).
  **Update, 2026-08-19: the BRAM overshoot is fixed.** The 1728-way fully
  unrolled `MAC` loop in `conv3d_hidden_layer` (`dz,dy,dx,ic` all unrolled
  for the `IN_C=64` hidden layer) was the dominant driver, not the frame
  buffer. De-unrolling the `ic` dimension (27-way spatial unroll kept,
  `ic` reduced serially over a pipelined loop) cut BRAM_18K usage to
  **114 / 4,032 (2.8%)**, comfortably under budget, verified bit-exact
  against `src/model/golden.py` via csim before resynthesizing. The cost:
  worst-case latency cycle count went up ~31.8x (48,077 -> 1,528,325
  cycles), on top of an already-bad number, and the target clock used for
  this second run (v++'s own default, 3.33 ns / ~300 MHz) is not the same
  as the first run's explicit 100 MHz target, so the millisecond figures
  are not directly comparable between the two rows; see
  `docs/DECISIONS.md` for the full numbers and caveats. Closing the
  latency gap (shot-level pipelining, further resource/latency tuning) is
  still follow-up work, not yet done.
- **Update, 2026-08-19: a real, timing-closed `.xclbin` was built.** First
  attempt (`v++ -l -t hw` at v++'s default ~300 MHz target, 3h54m) produced
  a well-formed but timing-unclosed bitstream (WNS -0.609 ns, 8,418 of
  701,750 endpoints failing). Rebuilt the link step only (same `.xo`, no
  need to resynthesize) at an explicit 100 MHz target via
  `hls/link_100mhz.cfg`, 2h33m: **WNS +0.003 ns, TNS 0.000 ns, 0 failing
  endpoints -- timing closed**, margin thin (3 ps) but real. Kernel-scoped
  post-route utilization (`results/metrics/hw_build_int8.json`, from
  Vivado's `report_accelerator_utilization`, excludes the platform shell):
  LUT 8.57%, FF 3.42%, BRAM 4.29%, DSP 3.67% of the kernel's user budget --
  all comfortably low. **Real latency at the actual configured 100 MHz
  clock: 13.04-15.28 ms per shot** (cycle counts unchanged from the HLS
  estimate, only the clock changed), roughly 15,000x the paper's own ~1us
  aspirational target; closing that gap is still real follow-up work. This
  is the first data point in the whole project that is simultaneously
  within the device's resource budget and timing-closed -- i.e. an honest
  candidate for an actual on-card run, though that run itself has not been
  attempted (`host/run_hw.py`'s real-hardware path remains an unexercised
  `NotImplementedError` stub; see the Phase 5 entries below). Full numbers
  and the BRAM-fix-then-timing-fix trail: `docs/DECISIONS.md`.
- **Cosim (real RTL simulation via XSIM) passed: 10/10 shots bit-exact**,
  confirming the synthesized gate-level design, not just the C++ model,
  matches `src/model/golden.py`. See `docs/DECISIONS.md`, 2026-08-18. The
  `hw` bitstream build has not been run (gated behind `BUILD_HW=1`, not
  requested).

## Phase 5: host integration

- **No real hardware run has happened.** `host/host_predecoder.cpp` and
  `host/run_hw.py` are written against the real XRT/pyxrt APIs (the C++ host
  compiles and links against the real `libxrt_coreutil.so` on this host,
  confirming API correctness), but neither has run against actual hardware:
  no xclbin exists for `predecoder_kernel` (the `hw` build was not
  attempted; see the Phase 4 entries in `docs/DECISIONS.md`). `run_hw.py`'s
  "no xclbin available" fallback to the golden model is the only path
  exercised for real, and is clearly labeled `"simulated": true` in every
  result it writes; never silently. `run_on_hardware()` (the real-hardware
  decode/timing path) raises `NotImplementedError` rather than pretending
  to work, since it has never actually run.
- **End-to-end latency numbers in the simulated path only cover the model
  forward pass and PyMatching decode**, not host-to-device / kernel /
  device-to-host transfer (those fields are `null`, not estimated or
  invented, since there is no real transfer happening in the fallback
  path). Real transfer numbers require an actual hardware run.
