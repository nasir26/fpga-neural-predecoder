# Decisions Log

## 2026-08-16: venv Python version, 3.11 vs 3.8

- **Decision:** use Python 3.8.10 (the pre-existing `python3` on this host) for
  the project venv, not python3.11.
- **Alternatives considered:** python3.11 is present at `/usr/bin/python3.11`
  and has better-maintained current wheels for stim/pymatching/torch, but its
  `venv` module is broken here: `ensurepip` is missing because the
  `python3.11-venv` apt package is not installed, and installing it requires
  `sudo`, which needs a password not available in this session.
- **Reason:** python3.8's venv works immediately with no privileged access
  needed, keeping Phase 0 unblocked. Asked the user; they chose the 3.8
  fallback over pausing to get sudo access.

## 2026-08-16: stim pinned to 1.12.1, not the latest 1.16.0

- **Decision:** pin `stim==1.12.1` in requirements.txt.
- **Alternatives considered:** stim 1.16.0 (latest at resolution time) has no
  prebuilt wheel for cp38 and must build from source; its C++ extension
  requires `-std=c++20`. This host's g++ is 9.4.0 (Ubuntu 20.04 default),
  which only supports up to `-std=c++17`/`-std=c++2a` partial support, not a
  full C++20 mode, so the build fails with
  `error: unrecognized command line option '-std=c++20'`. No newer g++
  (10/11+) binary is installed, and installing one needs `sudo`.
- **Reason:** stim 1.12.1 is the newest release whose C++ extension still
  targets `-std=c++17` (verified by inspecting `setup.py` across 1.10.0
  through 1.16.0) and it ships a prebuilt manylinux cp38 wheel, so no local
  compilation is needed at all. This is a real deviation from "use the latest
  stim" and is recorded here plus in requirements.txt and
  `docs/LIMITATIONS.md` (to be added when Phase 1 starts writing that file).

## 2026-08-16: hardware/toolchain inventory found on this host

- **Finding, not yet a decision requiring a tradeoff, but recorded for later
  phases:** this machine unexpectedly already has a real Xilinx Alveo U55C
  card installed (`lspci` shows Xilinx device 505c/505d at `8c:00.0/.1`),
  XRT 2.15.225 (branch 2023.1) live at `/opt/xilinx/xrt` with the device
  reporting `Device Ready: Yes`, the matching platform
  `xilinx_u55c_gen3x16_xdma_3_202210_1` installed under `/opt/xilinx/platforms`,
  a working `pyxrt` cp38 binding at `/opt/xilinx/xrt/python`, and two Vitis HLS
  installs (`/tools/Xilinx/Vitis_HLS/2021.2` and `/2023.2`), both of which run
  without an immediate license prompt.
- **Implication for later phases:** Phase 4/5 planning should not assume
  simulation-only is the ceiling; a real `hw` build and on-card run may be
  possible on this host. This does not change the Phase 0-3 scope or the
  "no bitstream without being asked" rule; it is recorded here so the choice
  of Vitis HLS version (2021.2 vs 2023.2) can be made deliberately when
  Phase 4 starts, matching XRT 2023.1 and the installed platform file.

## 2026-08-16: dropped "all four orientations" to X/Z basis only

- **Decision:** implement only `rotated_memory_x` and `rotated_memory_z`
  (Stim's two native rotated surface code tasks), not four orientations.
- **Alternatives considered:** hand-building two additional rotated/reflected
  circuit variants; adding 90-degree grid rotation as data augmentation in
  Phase 2 training instead.
- **Reason:** `stim.Circuit.generated` has no third/fourth built-in
  orientation, and the rotated surface code's boundary symmetry means a
  hand-built third/fourth variant is physically equivalent to X/Z memory
  under relabeling. Asked the user; they chose to drop the requirement
  rather than take on hand-built circuit risk for no new physical content.
  Logged in `docs/LIMITATIONS.md`.

## 2026-08-16: deferred the 25-parameter noise model to uniform-depolarizing only

- **Decision:** `NoiseModel` (`src/qec/circuits.py`) implements only
  `kind="uniform_depolarizing"` (a single `p` mapped onto Stim's four
  built-in noise knobs). `kind="full25"` is defined with all 25 field names
  reserved but raises `NotImplementedError`.
- **Alternatives considered:** hand-constructing the full circuit-level noise
  model now, with independent SPAM, idle, and all 15 two-qubit Pauli-channel
  probabilities injected via `PAULI_CHANNEL_1`/`PAULI_CHANNEL_2`.
- **Reason:** the full model needs a from-scratch circuit builder instead of
  `stim.Circuit.generated`, a much larger scope with real correctness risk
  (a subtle bug there would silently corrupt every downstream LER number).
  Asked the user; they chose to defer it and keep Phase 1 on the
  well-tested generator path. Logged in `docs/LIMITATIONS.md`.

## 2026-08-16: detector grid mapping and diff-frame convention

- **Decision:** the dense decoding volume has shape `(shots, T, H, W)` with
  `T = rounds + 1`, `H = W = distance + 1`, built by dividing Stim's
  even-integer detector `(x, y)` coordinates by 2. Verified empirically
  (`d=3,5` at both bases) that this mapping is injective and its cell count
  equals `circuit.num_detectors` exactly, so unused grid cells (the rotated
  code's checkerboard gaps, and the half-density boundary layers at `t=0` and
  `t=rounds`) are simply always zero.
- **Decision:** `diff_frames` (`src/qec/sampling.py`) XORs consecutive
  T-layers of the volume, with `diff[:,0] = volume[:,0]` (no earlier frame to
  difference against) rather than dropping the first frame or zero-padding
  it. This keeps `diff_frames` the same shape as `volume`, which matters for
  Phase 2's `distance = rounds = R` receptive-field convention.
- **Reason:** these are implementation choices, not requirements stated
  verbatim in the plan; recorded here so Phase 2's model input shape
  assumptions have a documented source.

## 2026-08-16: R=9 training deferred until R=5 validates the pipeline

- **Decision:** run the full spec-default training recipe (2e6 shots/epoch,
  20 epochs, batch 256, width 64) at R=5 (depth 2) now; defer R=9 (depth 4)
  sizing until R=5 results are in hand.
- **Alternatives considered:** shrinking the R=9 recipe now (fewer
  epochs/shots/width); running the full R=9 recipe as an hours-long
  background job immediately.
- **Reason:** measured on this CPU (all 64 threads already used by default),
  one training step is 37ms/batch at R=5 versus 438ms/batch at R=9. At the
  spec's batch count (~7800 batches/epoch), that is roughly 1.7 hours total
  for R=5 versus roughly 20 hours for R=9. This also appears to be a shared
  machine (another process already on the one GPU present). Asked the user;
  they chose to validate correctness cheaply on R=5 first rather than
  committing ~20 hours of CPU time to an unvalidated pipeline.

## 2026-08-16: ground-truth training target is XOR-parity of local fired DEM errors, not OR

- **Decision:** the per-voxel training target is the XOR-parity (not
  logical OR) of DEM error mechanisms that (a) actually fired in that shot
  and (b) touch at most 2 detectors ("local" errors; see
  `src.qec.dem.build_error_incidence_matrix`).
- **Alternatives considered:** OR of all fired-error footprints, which is
  simpler but was found to be subtly wrong: if two fired errors happen to
  toggle the same detector, their contributions cancel under XOR (the actual
  detector semantics) but OR would still mark that voxel as a target,
  disagreeing with the real detection volume in that rare case. Including
  all fired errors (not just weight <= 2 ones) in an XOR-parity target was
  also considered, but that reduces exactly to the raw detection volume
  itself (trivial, not a meaningful "local correction" target).
- **Reason:** XOR-parity restricted to local errors is self-consistent
  (never disagrees with the true detector semantics) and matches the
  method's framing of removing a large fraction, not all, of detection
  events; nonlocal (weight >= 3) fired errors are deliberately left for
  PyMatching. Caught while implementing, not asked about; the fix follows
  directly from how Stim defines detectors, so it is a correctness fix
  rather than a preference tradeoff.

## 2026-08-16: PreDecoder3D output head is a 1x1x1 conv, not a 3x3x3 conv

- **Decision:** the final channel-projection layer of `PreDecoder3D`
  (`src/model/predecoder.py`) uses `kernel_size=1`, not the `kernel_size=3`
  used by the hidden layers.
- **Alternatives considered:** kernel_size=3 for every layer, matching the
  plan's blanket "kernel 3" description literally.
- **Reason:** with a kernel-3 output layer, a depth-D stack of kernel-3
  hidden layers plus a kernel-3 output layer gives receptive field
  `2*D + 3`, not the plan's explicitly stated depth-2-gives-R=5,
  depth-4-gives-R=9 correspondence (`2*D + 1`). Verified empirically via
  input-gradient support before and after the fix (see
  `tests/test_predecoder.py`). A 1x1x1 output head is also consistent with
  the plan's own "output head" terminology (channel projection only, no
  added spatial context).

## 2026-08-16: R=5 trained model makes LER worse, not better; root cause diagnosed

- **Finding:** after training R=5 to completion (20 epochs, loss
  0.03198 -> 0.02162, `results/metrics/train_R5.json`) and evaluating with
  `bench/run_predecoder.py` at d=5, p=0.003 (`results/metrics/predecoder_d5_R5_p0.003.json`),
  the pre-decoder's syndrome density reduction is real (SDR ~0.62, it does
  remove a majority of detection events) but logical error rate gets much
  worse: LER 0.160 with the pre-decoder versus 0.00291 for PyMatching alone
  on comparable shots (about 55x worse), and this holds even at a
  same-shots apples-to-apples comparison (0.0036 baseline vs 0.164 to 0.079
  across thresholds 0.5 to 0.99).
- **Root cause, established by direct experiment, not guessed:** the model
  is well calibrated per voxel (mean predicted probability 0.648 where a
  real detection event is present versus 0.0019 where it is not, and the
  false-positive rate at threshold 0.5 is already only 0.043%), so the
  per-voxel classification signal itself is real. Restricting corrections to
  only ever cancel voxels that are already 1 in the raw syndrome (never
  inventing new detection events) barely changes the result (LER still
  0.129 to 0.157 across thresholds). This isolates the harm to something
  other than false positives: independent per-voxel thresholding can
  correctly identify that a real local fault occurred but cancel only one
  endpoint of its (typically two-detector) footprint while leaving the
  other untouched, breaking the parity structure PyMatching's minimum-weight
  matching depends on. A "mostly correct" per-voxel prediction can still be
  actively harmful once applied via independent-voxel XOR, because the
  matching graph is defined on edges (pairs), not on individual voxels.
- **Status:** resolved by switching to the per-local-edge output head; see
  the next entry. The R=5 results cited above (`train_R5.json`,
  `predecoder_d{5,7,9,11}_R5_p0.003.json`) were from the old per-voxel
  architecture and have been deleted from `results/metrics/` since they no
  longer correspond to any code in the repository; they are preserved in
  git history (commit `15cfb2d`) if needed for comparison.

## 2026-08-16: switched to a per-local-edge output head (fixes the parity bug)

- **Decision:** `PreDecoder3D` now outputs `1 + len(edge_offsets)` channels
  instead of 1: channel 0 for weight-1 ("singleton") local DEM errors,
  channels 1..N for the N distinct relative offsets among weight-2
  ("paired") local DEM errors. `edge_offsets` is computed dynamically from
  the training circuit's DEM (`src.qec.dem.build_local_edge_catalog`), not
  hardcoded, and stored in the model checkpoint so evaluation at any
  distance reuses the exact catalog the model was trained with. Corrections
  are applied as complete units
  (`src.model.edges.apply_local_edge_corrections`): a paired channel's
  activation always flips both its anchor and offset-partner voxel
  together, so a real fault's footprint can never be half-cancelled.
- **Alternatives considered:** post-processing a pair-agreement gate on the
  existing per-voxel model (cheaper, no retraining, but heuristic and the
  model was never trained to produce pairable outputs); a parity-aware loss
  term on the same per-voxel architecture (keeps the simpler architecture
  but only discourages, doesn't structurally prevent, half-cancellation).
- **Reason:** asked the user after diagnosing the root cause; they chose
  the structural fix over the two heuristic ones. The offset catalog was
  verified empirically stable across d=5,7,9,11 (Z basis) and checked at
  d=9 for X basis (13 offsets each, with one offset differing between X and
  Z as expected from the code's transpose duality) before committing to
  this design, so it is safe to treat as a fixed-size, distance-independent
  channel set.
- **Consequence:** the R=5 training run must be redone from scratch with
  the new architecture (previous checkpoint and results deleted; see
  previous entry). Retrained; see the next two entries for what was found.

## 2026-08-16: second bug found by re-evaluating the retrained model: local corrections silently dropped their own observable-flip effect

- **Finding:** after retraining R=5 with the edge-based architecture (loss
  0.00435 -> 0.00222 over 20 epochs) and evaluating, LER at d=5, p=0.003 was
  0.106, still about 36x worse than baseline (0.00291), essentially
  unchanged from the old architecture's failure mode. This ruled out "half
  cancelled pairs" as the (whole) explanation, since the new architecture
  makes that structurally impossible.
- **Root cause, confirmed with an oracle experiment before touching
  production code:** applying a correction (whether from the model or, to
  isolate the question, from perfect ground-truth local edges) removes that
  edge's contribution from the syndrome PyMatching sees, but does nothing to
  account for whether that same edge also flips the logical observable.
  Measured directly from the DEM at d=5, p=0.003: 82 of 696 local errors
  (about 12%) carry an observable flip. An oracle-correction test using the
  real fired local edges but with NO observable tracking reproduced the
  exact same catastrophic failure (LER 0.117 vs baseline 0.00305) as both
  architectures' trained models. This proved the problem was never about
  model accuracy; it was a missing term in the evaluation pipeline itself.
- **Fix, validated before applying to production code:** the same oracle
  experiment, extended to also XOR the local corrections' own observable
  parity into PyMatching's residual-decode result before comparing to the
  true observable, gave LER 0.00115, *better* than baseline (0.00305), not
  worse. Implemented as `src.qec.dem.build_channel_observable_map`
  ((channel, anchor detector) -> whether that local edge flips the
  observable, built from the same DEM the catalog and incidence matrix come
  from) and `bench.run_predecoder.local_correction_observable_parity`
  (applies it to whatever edges were actually corrected, model or oracle).
  This only affects evaluation, not training: the training loss never
  touched observables, so no retraining was needed for this fix.

## 2026-08-16: with both bugs fixed, the trained R=5 model still isn't net-positive at any single decision threshold

- **Finding:** with the observable-tracking fix applied, LER at d=5,
  p=0.003, threshold=0.5 improved to 0.0897 (from 0.106), but is still
  about 31x worse than baseline. A threshold sweep (0.5 to 0.999) shows LER
  improves as threshold rises (0.090 -> 0.078 -> 0.064 -> 0.056 -> 0.0038
  -> 0.0035 at thresholds 0.5, 0.7, 0.9, 0.95, 0.99, 0.999) but syndrome
  density reduction collapses in the same direction (0.55 -> 0.50 -> 0.37
  -> 0.26 -> 0.009 -> 0.000): the model only stops being net-harmful once
  it is so conservative it is barely doing anything. At every threshold
  tried, LER never drops meaningfully below the 0.00291 baseline while SDR
  is still substantially positive; there is no operating point in this
  sweep where the pre-decoder is both useful (removes a real fraction of
  detection events) and net-beneficial (LER same or better than baseline).
  The formulation is now provably sound (the oracle test above beats
  baseline), so this is a model-precision problem, not a pipeline bug: the
  trained model has not learned to distinguish real local edges from
  plausible-looking false ones precisely enough, given the current 20-epoch,
  2e6-shots/epoch, width-64 training budget. Flagged to the user rather than
  quietly accepted or further tuned without checking in; see the
  conversation for how this was resolved.

## 2026-08-16: third bug found: build_channel_observable_map double-counted an even-parity observable target as a flip

- **Finding:** the threshold-sweep re-diagnosis above used
  `build_channel_observable_map`, which turned out to have its own bug,
  found while writing a regression test for the *previous* fix (the test
  initially failed: oracle-correction LER was 0.104, not competitive with
  baseline, even with observable tracking supposedly applied).
- **Root cause:** 25 of the local DEM error lines at d=5, p=0.003 list the
  logical observable target twice (e.g. a merged error line combining two
  underlying mechanisms that each individually flip the observable, netting
  to no flip). `build_channel_observable_map` treated "the observable
  appears in this error's target list at all" as "this error flips the
  observable," rather than computing XOR-parity of how many times it
  appears. This inflated the map from the correct 53 nonzero (channel,
  anchor) entries to 78, causing the composed prediction to be wrong for
  roughly a quarter of all observable-carrying local edges.
- **Fix:** `build_channel_observable_map` now flips only when
  `len(obs_ids) % 2 == 1`. Verified directly (obs_map.sum() now 53,
  matching hand computation) and via the regression test in
  `tests/test_observable_tracking.py`, which was failing before this fix
  and passes after it.
- **Result after both observable-tracking bugs were fixed:** re-evaluating
  the same R=5 checkpoint at threshold=0.5, LER at d=5, p=0.003 improved
  further to 0.0129 (from 0.0897), about 4.4x worse than baseline (0.00291),
  down from the original 55x. A threshold sweep at d=5 shows a genuinely
  promising operating point around threshold=0.95: LER 0.00303 (within
  noise of baseline's 0.00291) while still achieving 26% syndrome density
  reduction. LER still degrades with distance beyond the R=5 training
  distance (d=7: 0.066, d=9: 0.104, d=11: 0.156, all worse than baseline
  where available), meaning the distance-generalisation property is not
  yet demonstrated. This is the honest state of Phase 2 as of this entry;
  not yet resolved further.
- **Separate, non-correctness issue noticed while investigating:** the
  per-shot timing loop in `bench/run_predecoder.py` reports absurd
  mean_latency_us_per_shot values (>100,000us) when `timing_shots=1000` and
  `shots=100000`, but reports normal values (~1-2ms) at smaller
  `timing_shots`. Isolated micro-benchmarks of every component (model
  forward, matcher.decode, edge correction, observable parity) run in
  ~1-2ms combined, so this is a benchmarking-harness bug (likely something
  degrading over many loop iterations, not yet root-caused), not a real
  latency number. Latency numbers in the committed `predecoder_d*.json`
  files should not be trusted until this is fixed; LER, SDR, and
  ler_reduction_factor are unaffected and are correct. Flagged for a later
  pass (Phase 6 benchmarking, or sooner if latency numbers are needed
  before then).

## 2026-08-17: timing loop measured thread contention, not real latency

- **Root cause found:** two duplicate `bench/run_predecoder.py` background
  processes ended up running simultaneously (an artifact of this session's
  background-job handling, not the code), pushing system load to ~92-99 on
  a 64-core machine. Under that load, re-running the exact same evaluation
  in isolation gave normal timing (967us/shot, matching component-level
  micro-benchmarks); the duplicate-process run gave >100ms/shot. Confirmed
  by killing the duplicates and re-running cleanly.
- **Underlying issue, fixed regardless:** even without duplicate processes,
  the single-shot timing loop was letting PyTorch spin up its full 64-thread
  parallel region for every single tiny (batch=1) model forward call. A
  real single-shot deployment would never do this (64-way parallelism for
  one small tensor is pure overhead, and it is also what made the loop so
  vulnerable to contention from anything else running concurrently).
  `bench/run_predecoder.py`'s timing loop now pins `torch.set_num_threads(1)`
  for its duration (restored afterward), which both fixes the contention
  sensitivity and makes the measured number mean what it claims to mean:
  realistic single-shot latency, not artificially-parallelized-then-starved
  latency.

## 2026-08-17: bias must be quantized too, and "zero tolerance" means the decision, not the float bits

- **Finding while building golden.py:** the first cut of `QATPreDecoder3D`
  kept biases at full float precision (reasoning: a small per-channel
  additive term, cheap to keep wide in hardware). That is fine for a real
  accumulator with enough guard bits, but it makes `golden.py` (a genuine
  integer datapath) unable to match the QAT model's forward pass at all,
  since an integer accumulator cannot add an arbitrary-precision float.
  Fixed by quantizing bias to each layer's accumulator scale
  (`weight_scale_c * input_scale`) in both `src/model/quantize.py`
  (`quantize_bias`/`fake_quantize_bias`) and `src/model/qat.py`'s forward
  pass, so both sides use the same integer bias.
- **Finding while writing test_golden_vs_torch.py:** even with matching
  integer bias handling and even upcasting the QAT model to float64,
  `golden.py`'s NumPy integer convolution and PyTorch's `F.conv3d` do not
  produce literally bit-identical floats (residual ~1e-7 relative at the
  real R5 model's width=64 scale). This is floating-point non-associativity:
  torch's BLAS-based convolution and golden.py's tensordot-based
  accumulation sum the same terms in a different order, and float addition
  is not associative. No amount of added precision removes this; only
  reimplementing QAT training itself in pure-integer torch tensors would
  (abandoning `F.conv3d` and the batched-training speed it gives).
- **Decision:** "zero tolerance" is enforced on the correction *decision*
  (sign of the final accumulator: exactly what gets applied to the
  syndrome), which is genuinely zero-tolerance and is what correctness
  actually depends on. The underlying float values are compared at 1e-6
  relative/absolute tolerance (observed residual is ~1e-7 at real model
  scale, so this has a comfortable margin) rather than literal bit-equality.
  Asked the user rather than silently picking an interpretation of "zero
  tolerance" that conflicts with how floating-point arithmetic actually
  works; they chose this over rewriting QAT training in pure-integer torch.

## 2026-08-17: Phase 3 quantization sweep results (contribution C1)

- QAT fine-tuning: 3 epochs, 500,000 shots/epoch, from the R=5 FP32
  checkpoint, one run per precision (int8, int4, ternary). Final losses:
  int8 0.00221, int4 0.00238, ternary 0.00271 (fp32's own final training
  loss was 0.00222; all three stayed close to it).
- Sweep at d=5,7, p=0.003, threshold=0.5, 100,000 shots (`bench/sweep_precision.py`,
  `results/metrics/quantization_sweep.json`):

  | precision | d=5 LER | d=5 SDR | d=7 LER | d=7 SDR |
  |---|---|---|---|---|
  | fp32     | 0.01292 | 0.550 | 0.06593 | 0.507 |
  | int8     | 0.01258 | 0.552 | 0.06515 | 0.508 |
  | int4     | 0.01424 | 0.548 | 0.07124 | 0.504 |
  | ternary  | 0.01303 | 0.526 | 0.06435 | 0.486 |

  INT8 is within noise of FP32 (slightly better, if anything: QAT
  fine-tuning's extra gradient steps likely account for this more than
  quantization itself). Ternary is close to FP32 on LER with a real SDR
  drop. INT4 is the only precision with a clear, consistent LER regression
  (~8-10% relative at both distances). This is a genuine, clean C1 result
  in its own right: quantization down to INT8, and even ternary, costs
  little to nothing relative to the FP32 baseline, even though that FP32
  baseline itself does not yet beat plain PyMatching (see the Phase 2
  entries above); the two findings are independent and both honestly
  reported.
- HLS weight headers and binary blobs exported for all three precisions to
  `hls/weights_{precision}.h` / `.bin`, with SHA-256 in both the header
  comment and `results/metrics/quantization_sweep.json`.

## 2026-08-17: HLS kernel scoped to a fixed distance (d=5)

- **Finding:** the observable-parity correction (`obs_map`, see the Phase 2
  observable-tracking entries above) is not a fixed, distance-independent
  quantity in the way the local-edge offset catalog is. It IS geometrically
  regular (empirically, the observable-flipping positions are always within
  rows 0-2, near one fixed physical code boundary, at every distance
  checked: d=5,7,9), but confirming the *exact* general rule well enough to
  hardcode it for arbitrary runtime distances would need more
  reverse-engineering, with real risk of another subtle bug on top of
  everything else found this project.
- **Decision:** the HLS kernel (`hls/predecoder_kernel.cpp`) is scoped to a
  single fixed compile-time distance, d=5 (T=H=W=6), with the
  observable-parity lookup table (`hls/kernel_config.h`'s
  `OBS_PARITY_LUT`, 14 x 6 x 6 x 6, only 53 nonzero entries) embedded
  directly as a constant, computed by `scripts/export_kernel_config.py`
  from the same, already-tested `build_channel_observable_map`. The CNN
  datapath itself (`conv3d_hidden_layer`, `line_buffer.h`) remains
  architecturally distance-general (a conv layer has no inherent distance
  dependence); only the observable-parity table is distance-specific.
  Asked the user rather than silently choosing; they preferred the
  lower-risk fixed-distance scope over deriving and trusting an
  under-verified general rule. Re-deriving the table for a different
  distance is cheap (`python scripts/export_kernel_config.py`) if ever
  needed; a genuinely distance-general kernel would need either the general
  geometric rule or a way to reload a distance-specific table at
  configuration time, and is left as a documented limitation.
- **Consequence:** given a fixed d=5, T*H*W = 216 bits, comfortably fitting
  in a single 512-bit AXI beat; the top-level kernel reads/writes exactly
  one beat per shot rather than needing burst-oriented multi-beat packing.

## 2026-08-17: found and fixed a silent double-truncation bug in export.py

- **Finding:** `_format_c_array` (used to write both integer and `double`
  constexpr C arrays) called `str(int(v))` unconditionally regardless of
  the requested `dtype`, silently truncating every fractional value (weight
  scales, requantization multipliers, e.g. 0.4537, 1.5406) to a bare
  integer (0, 1, 2, ...) whenever it wrote a `double` array. This existed
  since Phase 3's original `export_hls_header` and was never caught then,
  because nothing consumed the corrupted `WEIGHT_SCALE`/`REQUANT_SCALE`
  arrays for real computation until the HLS kernel actually needed them.
  Caught immediately once the kernel was tested against golden.py-generated
  vectors: layer0's output diverged even for a window that should have been
  provably all-zero (i.e. only the bias path was exercised), narrowing the
  bug to bias/requant handling and then to the corrupted scale constants
  directly.
- **Fix:** `_format_c_array` now uses `repr(float(v))` for `dtype="double"`
  (Python's shortest round-tripping float representation) instead of
  truncating. Verified the fix directly (re-exported headers show correct
  fractional values, e.g. `1.5406237177601014`) and with a regression test
  (`tests/test_export.py::test_double_arrays_preserve_fractional_precision`).
- **Scope of the corruption:** only the `.h` header's literal text was
  affected. The `.bin` binary blob (and therefore its SHA-256) never
  included the scale arrays, so `results/metrics/quantization_sweep.json`'s
  recorded SHA-256 values, and every LER/SDR number reported for Phase 3,
  were computed in Python directly from `golden.py`/`QATPreDecoder3D` and
  were never affected by this bug. Re-exporting after the fix reproduced
  identical SHA-256 hashes, confirming this.

## 2026-08-17: HLS kernel verified bit-exact against golden.py (200/200 vectors)

- After the fixes above, `hls/predecoder_kernel.cpp` (line-buffered 3D conv
  x2, 1x1x1 output conv, edge-correction + observable-parity stage) matches
  `src/model/golden.py`'s output exactly (correction bits and the
  observable-parity bit) on all 200 committed test vectors
  (`tests/vectors/vectors_int8.txt`, generated by
  `scripts/export_test_vectors.py`), verified via a standalone g++ compile
  of `hls/tb_predecoder.cpp` (fast iteration; full Vitis HLS csim/csynth is
  the next step).

## 2026-08-17/18: two real csynth runaways, both from over-aggressive pragmas, both fixed

- **First runaway:** the initial kernel fully unrolled both the per-voxel
  output-channel loop (`OC_LOOP`, 64-way) and the inner MAC reduction
  (kernel positions x input channels, up to 27*64=1728-way for layer 1),
  compounding to a ~110,592-way fully-parallel MAC unit demanded for a
  single II=1 pixel. csynth's "Compiling Optimization and Transform" phase
  alone ran past 3800 CPU-seconds (over an hour of wall time) without
  finishing; killed after confirming via `ps` it was still actively
  computing (not hung) but clearly not converging in reasonable time. Fixed
  by moving `#pragma HLS PIPELINE II=1` from the per-pixel (y,x) loop down
  to the `OC_LOOP` itself: output channels are now computed sequentially
  (pipelined at II=1 per channel, not all 64 in parallel), while the MAC
  reduction stays fully unrolled per channel. Verified this didn't change
  results (pragmas are no-ops for a plain g++ build; re-ran the 200-vector
  g++ testbench, still bit-exact) before re-attempting csynth. The retry's
  "Compiling Optimization and Transform" phase dropped to ~315s, then
  ~180s: real confirmation the fix worked, not a guess.
- **Second runaway:** with the first fix in place, csynth reached
  `edge_correction`'s main loop and again ran for hours (crossed midnight;
  killed after seeing 775+ CPU-minutes with no forward progress signal).
  Root cause: `correction[T][H][W]` was fully array-partitioned (216
  individual registers) AND accessed with *data-dependent* indices
  (`correction[tt][yy][xx]` where tt/yy/xx depend on which local-edge
  channel fired) AND the enclosing loop demanded II=1. That combination
  forces the tool to build a full-width mux for every one of ~2800
  variable-indexed writes per shot, with additional cross-iteration
  read/write-hazard analysis, since a later pipelined iteration can target
  the same storage a concurrently in-flight iteration wrote. Fixed by
  removing the array partition (letting `correction` be a normal
  BRAM-mapped array) and removing the forced II=1 (edge_correction runs
  once per shot and is not the throughput bottleneck; the conv layers are).
  The retry succeeded, and the resulting *natural* schedule the tool found
  is Final II=38 for that loop (not II=1, but converged in about 2
  minutes, not hours) -- a slower, but real and immediately obtained,
  answer, which is the right tradeoff here.
- **General lesson, recorded because it will recur if forgotten:** forcing
  II=1 on a loop is only free when the loop's body has no cross-iteration
  dependency on shared, indexable storage. Where it does (variable-indexed
  reads/writes to a partitioned array), forcing II=1 anyway does not fail
  cleanly -- it can make the scheduler search for an exponentially long
  time before either succeeding or (in practice, within any reasonable
  session) not finishing at all. The fix in both cases was the same shape:
  push aggressive pipelining down to a loop level where the parallelism is
  actually free (independent output channels; independent kernel-position
  MACs), and let the tool pick a realistic, unforced II wherever iterations
  can genuinely conflict.

## 2026-08-18: csynth completed; real (uneven) resource and latency numbers

- `hls/predecoder_kernel.cpp`, INT8, target `xcu55c-fsvh2892-2L-e`
  (Alveo U55C part), `create_clock -period 10` (100 MHz target): "All loop
  constraints were satisfied", Estimated Fmax 136.99 MHz (i.e. the design
  can run faster than the 100 MHz target). Parsed automatically from the
  csynth XML by `scripts/record_hls_metrics.py` into
  `results/metrics/hls_int8.json` (never hand-transcribed):
  - Latency: 35,336 cycles best/avg case, 48,077 cycles worst case (0.353 ms
    / 0.481 ms at the 100 MHz target clock).
  - DSP 1,219 / 9,024 (13.5%), FF 254,875 / 2,607,360 (9.8%),
    LUT 428,939 / 1,303,680 (32.9%), URAM 0 / 960 (0%).
  - **BRAM_18K: 46,716 used against 4,032 available -- 1159%, over 11x the
    real budget.** This design, as synthesized, would not fit on the actual
    device. Recorded honestly, not hidden: this is a first-pass,
    correctness-prioritized kernel (see the pragma-runaway entries above),
    not yet resource-tuned. The BRAM overshoot traces to weight/window array
    partitioning decisions made to support the (already reduced from
    110,592-way) per-channel MAC parallelism; bringing it under budget is
    follow-up work (narrower partitioning, more resource sharing across
    output channels), not yet done.
  - **Latency is far from the paper's own ~1us aspirational target**
    (0.35-0.48 ms per shot, roughly 350-500x over): the kernel currently
    processes one shot mostly-sequentially through 4 DATAFLOW stages with
    no shot-level pipelining (`DataflowPipelineThroughput` interval
    8,697-48,035 cycles). This is the honest, current state of contribution
    C4's latency budget, not a claimed result; the paper must say so
    plainly rather than imply the aspirational target was met.

## 2026-08-18: cosim PASS -- RTL simulation matches golden.py bit-exactly (10/10)

- `cosim_design` (real RTL simulation via XSIM, not the C model) ran all 10
  committed test vectors (`tests/vectors/vectors_int8_small.txt`) through
  the actual synthesized RTL and compared against the same golden.py-derived
  expected outputs `tb_predecoder.cpp` checks in csim. Result: **"C/RTL
  co-simulation finished: PASS"**, 10/10 shots bit-exact. This is the
  strongest correctness evidence available short of an actual bitstream on
  real hardware: the gate-level (post-synthesis) behavior, not just the
  high-level C++ model, matches the bit-exact NumPy golden reference.
- Runtime: cosim alone took about 39 minutes wall-clock (RTL simulation of
  ~48,000-90,000 cycles per shot across 10 shots, XSIM); the full
  `vitis_hls -f run_hls.tcl` invocation (csim + csynth + cosim together)
  took about 93 minutes total elapsed time. Progress reporting during cosim
  is per-shot (transaction), not continuous, which initially looked like a
  stall for the first ~95 seconds of each shot before its completion line
  printed; confirmed genuine (not stuck) by checking process CPU usage
  directly before waiting further.
- **Phase 4 status: csim, csynth, and cosim all complete and passing.** The
  `hw` bitstream build remains gated behind `BUILD_HW=1` and has not been
  run, per the plan's explicit instruction not to launch a multi-hour build
  without being asked.

## 2026-08-18: Phase 5 scoped to host code written-but-unexercised, fallback exercised for real

- **Decision:** `host/host_predecoder.cpp` (XRT C++: HBM bank assignment
  via `kernel.group_id()`, a depth-4 ping-pong/ring buffer of independent
  (input bo, output bo, run) slots for overlap across shots, per-stage
  `std::chrono` timing) and `host/run_hw.py` (the same design via pyxrt)
  are written to be correct against the real XRT API, but not run against
  real hardware: no xclbin exists for this kernel (the `hw` build was not
  attempted, given Phase 4's BRAM overshoot; see the Phase 4 entries
  above). `run_hw.py`'s "no xclbin available" fallback to the golden model
  IS exercised for real, since that is the actual, current state.
- **Verification done despite no xclbin:** `host_predecoder.cpp` was
  compiled AND linked (not just syntax-checked) against the real
  `libxrt_coreutil.so` on this host, confirming every API call
  (constructors, `group_id`, `bo::sync`, `run::start`/`wait`) matches the
  real library, not just the headers. It fails cleanly (informative error,
  no crash) when pointed at a nonexistent xclbin/shots file.
  `run_hw.py --shots 10000` was run for real: it prints a loud,
  un-missable warning, falls back to `QATPreDecoder3D` (the same model
  `golden.py` is built from), and labels the result `"simulated": true` in
  the committed JSON, with LER 0.0132 at d=5, p=0.003 -- consistent with
  Phase 3's int8 sweep result (0.01258-0.01424 depending on seed/shots),
  which is itself a useful cross-check that the fallback path is computing
  the right thing.
- **Alternatives considered:** attempting the `hw` build first so the real
  path could be genuinely exercised. Asked the user; they chose the
  lower-risk path (Phase 4's BRAM overshoot makes a `hw` build attempt
  likely to fail or run for a very long time before failing) over
  reopening Phase 4's resource-tuning work now.
- **Known gap:** `run_hw.py`'s real-hardware branch
  (`run_on_hardware` -> LER/latency aggregation) raises
  `NotImplementedError` if a real xclbin is ever found, rather than
  silently running incompletely-tested code path; finishing it is
  documented follow-up work for whenever a working bitstream exists.

## Post-Phase-5: revisiting the Phase 4 BRAM overshoot, attempting a real `hw` build

- **Trigger:** the user asked to actually build a real `.xclbin` via the
  standard `v++ -c -t hw` / `v++ -l -t hw` command-line flow (as distinct
  from Vitis HLS's `export_design -flow impl`, never attempted -- see
  Phase 4). Confirmed with the user before running: given the recorded
  1159%-of-budget BRAM overshoot, this was expected to very likely fail or
  run for hours before failing; the user chose to attempt it anyway
  (2026-08-18) for real, honest data rather than a guess.
- **`v++ -c -t hw` (first attempt, unmodified kernel), 2026-08-18:**
  succeeded in 1h28m, produced a 437MB `.xo`. All loop constraints
  satisfied, Estimated Fmax 360.23 MHz. This step runs Vitis HLS
  internally but does not itself do place-and-route, so it does not
  surface the BRAM overshoot as a hard failure; the link step
  (`v++ -l -t hw`, full Vivado place-and-route) was expected to be where
  it would actually fail. Not yet attempted at this point.
- **Root-caused the BRAM overshoot before attempting the link step.**
  Reading `hls/predecoder_kernel.cpp`'s `conv3d_hidden_layer` MAC loop
  (not `hls/line_buffer.h`'s `FrameBuffer3D`, an earlier and much smaller
  suspect that was proposed and explicitly not pursued) found the real
  driver: for the wide hidden layer (`IN_C=64`), the `MAC` loop had
  `dz,dy,dx,ic` **all** `#pragma HLS UNROLL`'d together -- 3x3x3x64 =
  1728-way fully parallel access into the 110,592-entry constant
  `weight_flat` array every cycle, with the offset depending on the
  runtime `oc` value. Serving 1728 simultaneous reads (only `oc` varying
  at runtime) forces the synthesizer to partition that weight array into
  ~1728 separately-addressable slices; if each lands on its own BRAM_18K
  primitive (plausible default, since Vitis HLS does not aggressively
  pack small unrelated slices into shared blocks), that is the right
  order of magnitude to explain 46,716 BRAM_18K instances.
- **Fix: de-unroll the `ic` dimension.** Restructured `conv3d_hidden_layer`
  (`hls/predecoder_kernel.cpp`) to keep only the 27 spatial taps
  (`dz,dy,dx`) unrolled into 27 partial accumulator registers
  (`#pragma HLS ARRAY_PARTITION variable=partial complete dim=0`), and
  added a new `IC_LOOP` that reduces over `ic` serially with
  `#pragma HLS PIPELINE II=1`, summing the 27 partials into the final
  `acc` after the loop. Since both the weight and window operands are
  integers accumulated into a fixed-width integer accumulator (`acc_t`),
  re-associating the summation this way is exact, not an approximation:
  integer addition is associative/commutative, unlike the FP32 case
  discussed in the Phase 3 entries above. Verified via a fast
  `csim`-only pass (`hls/run_csim_only.tcl`, new file, ~2 minutes, does
  not run synthesis) before paying for another full `v++` recompile:
  **10/10 shots still bit-exact** against `src/model/golden.py`, exactly
  matching the original Phase 4 cosim result.
- **`v++ -c -t hw` (second attempt, restructured kernel), 2026-08-19:**
  succeeded in 54m, `.xo` size dropped from 437MB to 13.5MB. Real
  synthesis numbers (`hls/_x/reports/predecoder_kernel/hls_reports/predecoder_kernel_csynth.rpt`,
  parsed into `results/metrics/hls_int8.json` by the now-extended
  `scripts/record_hls_metrics.py`, see below):

  | | Phase 4 (unmodified) | after MAC restructuring |
  |---|---|---|
  | BRAM_18K | 46,716 / 4,032 (1158.6%) | **114 / 4,032 (2.8%)** |
  | DSP | 1,219 / 9,024 (13.5%) | 276 / 9,024 (3.1%) |
  | FF | 254,875 / 2,607,360 (9.8%) | 288,347 / 2,607,360 (11.1%) |
  | LUT | 428,939 / 1,303,680 (32.9%) | 366,534 / 1,303,680 (28.1%) |
  | latency (worst case) | 48,077 cycles / 0.481 ms | 1,528,325 cycles / 5.094 ms |

  The BRAM fix is real and large (~410x reduction, from 11x over budget to
  comfortably under it). The cost is also real: worst-case latency cycle
  count went up ~31.8x, roughly matching the expected cost of serializing
  a 64-wide reduction that used to be free (1 cycle) and now takes ~64
  pipelined cycles per output channel. Latency was already far over the
  paper's aspirational target before this change; it is now much further
  over. This was the explicit, agreed trade (fit the device, accept worse
  latency) -- not a free win.
- **Caveat: the two rows above are not measured at the same target clock.**
  Phase 4's csynth ran through the `vitis_hls` project flow
  (`hls/run_hls.tcl`), which explicitly sets `create_clock -period 10`
  (100 MHz target). This `v++ -c -t hw` run did not pass any clock/config
  argument, so `v++` used the U55C platform's own default kernel clock,
  which the report shows as 3.33 ns target (~300 MHz), not 10 ns. The
  cycle counts (48,077 vs 1,528,325) are the fair, clock-independent
  comparison; the millisecond figures in the table above are each correct
  for their own run but are not directly comparable to each other without
  accounting for this. Pinning an explicit clock via a `v++ --config` file
  (matching the reference pattern the user shared, which uses a
  `[clock] freqHz=...` directive) is follow-up work if a clean
  apples-to-apples comparison is needed for the paper.
- **`scripts/record_hls_metrics.py` extended, not replaced.** The
  `v++ -c -t hw` flow's report directory
  (`hls/_x/reports/predecoder_kernel/hls_reports/`) contains only a text
  `.rpt`, no XML -- the XML report Phase 4's parser expected only exists
  under the `vitis_hls` project flow's `solution1/syn/report/` directory.
  Rather than hand-transcribing numbers from the `.rpt` (a hard rule of
  this project; see the file's own docstring), added `parse_csynth_rpt()`
  alongside the existing `parse_csynth_xml()`, selected by file extension,
  and an explicit source-path argument (`record_hls_metrics.py <precision>
  <path>`) since the two flows put their reports in different places.
- **`v++ -l -t hw` (place-and-route / bitstream link), 2026-08-19:**
  completed in 3h54m27s and produced a real, well-formed
  `hls/build_hw/predecoder_kernel.xclbin` (53,088,266 bytes; gitignored, not
  committed -- binary build artifact, trivially regenerable from source and
  these logs). Verified, not assumed: `xclbinutilinfo` confirms the
  `Kernels:` field lists `predecoder_kernel`, targets the correct platform
  (`xilinx_u55c_gen3x16_xdma_3_202210_1`), and the `predecoder_kernel_1`
  instance's `IP_LAYOUT`/`CONNECTIVITY` sections map `syndrome_hbm` to
  `M_AXI_GMEM0`/`HBM[0]` and `correction_hbm` to `M_AXI_GMEM1`, matching the
  kernel's actual `m_axi bundle=gmem0/gmem1` interface exactly. No `ERROR`
  or `CRITICAL WARNING` lines in `hls/build_hw/vpp_link.log`.
- **Caveat, load-bearing, not glossed over: timing did not close.** The
  routed design's timing summary
  (`hls/_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt`)
  reports **WNS = -0.609 ns, TNS = -1217.878 ns, 8,418 of 701,750 endpoints
  failing** (~1.2%, each individually a small violation). Hold timing is
  clean (WHS 0.002 ns, 0 failing endpoints). `v++`'s `hw` flow packages an
  `.xclbin` regardless of timing closure -- producing the file is not the
  same claim as "this bitstream is safe to run on real hardware at its
  configured clock." Running an unclosed-timing bitstream at full speed
  risks setup-time data corruption on the failing paths; this has NOT been
  attempted (see `host/run_hw.py`'s existing "no usable xclbin" fallback
  behavior in the Phase 5 entries above -- pointing it at this xclbin
  specifically has not been tried and is a separate, real decision, not
  something to do silently).
  Background-task tracking of this run was itself interrupted partway
  through (an unrelated harness/session hiccup, not a build failure); the
  build process itself was independent of that tracking loop and completed
  on its own, confirmed by checking the real OS process state and the log
  file directly rather than assuming either success or failure from the
  broken notification.
- **Decision: rebuild at 100 MHz (10 ns) to close timing.** Confirmed with
  the user (2026-08-19); chose the lower-clock rebuild over trying Vivado
  P&R directives at the original ~300 MHz target, and over stopping with
  the unclosed-timing bitstream as the final recorded result. Added
  `hls/link_100mhz.cfg` (`[clock] freqHz=100000000:predecoder_kernel_1`,
  kernel instance name confirmed from the first build's
  `xclbin.info`, not guessed) and reran only the link step -- the existing
  `.xo` from the earlier `v++ -c -t hw` did not need to be regenerated,
  since HLS cycle counts do not depend on link-time clock assignment.
  Old artifacts preserved as `*_unclosed_300mhz.*` /
  `vpp_link_attempt1_unclosed.log` before overwriting (gitignored, not
  committed, but kept locally for comparison).
- **`v++ -l -t hw` at 100 MHz, 2026-08-19: timing closed.** Completed in
  2h33m13s (faster than the first attempt's 3h54m, consistent with easier
  place-and-route at a more relaxed clock target). Real numbers from the
  routed design (`hls/_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt`):
  **WNS = +0.003 ns, TNS = 0.000 ns, 0 of 700,988 endpoints failing**; hold
  timing also clean (WHS +0.009 ns, 0 of 699,255 endpoints failing). The
  margin is thin (three picoseconds) but genuinely non-negative, and no
  `ERROR`/`CRITICAL WARNING` lines appear in `hls/build_hw/vpp_link.log`.
  `predecoder_kernel.xclbin` (51,609,344 bytes) verified well-formed via
  `xclbinutilinfo` the same way as the first build.
- **New, more authoritative resource numbers: kernel-scoped, post-route.**
  Vivado's `report_accelerator_utilization` (in
  `hls/_x/reports/link/imp/impl_1_kernel_util_routed.rpt`) reports usage
  for `predecoder_kernel_1` specifically, excluding the U55C platform
  shell -- a more precise, real-hardware number than the earlier
  device-wide HLS csynth estimate (which does not know about the shell's
  own resource reservation). Added `scripts/record_hw_build_metrics.py`
  (parses this report plus the timing summary, does not hand-transcribe)
  writing `results/metrics/hw_build_int8.json`:

  | resource | kernel-scoped, post-route, % of kernel's user budget |
  |---|---|
  | LUT | 8.57% (101,290 used) |
  | FF | 3.42% (83,565 used) |
  | BRAM | 4.29% (78 used) |
  | DSP | 3.67% (331 used) |

  These percentages are of the kernel's *user budget* (device resources
  minus the platform shell's own reservation), not of total device
  resources, so they are not directly comparable number-for-number to the
  device-wide percentages in `results/metrics/hls_int8.json` -- both are
  real, just scoped differently; see the `note` field in
  `hw_build_int8.json`.
- **Final honest latency at the actual configured clock (100 MHz, cycle
  counts unchanged from the HLS estimate since only the clock changed):
  worst case 1,528,325 cycles = 15.283 ms, best case 1,304,125 cycles =
  13.041 ms per shot.** This is the real, closed-timing number -- worse in
  absolute time than both the original Phase 4 kernel (0.481 ms, but
  11x over BRAM budget and never built as `hw`) and the first (unclosed)
  `hw` attempt at ~300 MHz (5.094 ms, but not safe to run). Of the three
  real data points collected across this BRAM-fix-then-timing-fix
  sequence, this is the only one that is simultaneously (a) within the
  device's resource budget and (b) timing-closed -- i.e. the only one
  that is an honest candidate for an actual on-card run. It remains
  ~15,000x the paper's own ~1us aspirational target; closing that gap
  (shot-level pipelining, resource/latency retuning of the now
  comfortably-under-budget design) is real follow-up work, not attempted
  here.
- **End state of this line of work:** a real, verified, timing-closed
  `.xclbin` for `predecoder_kernel` at d=5/int8 now exists locally
  (`hls/build_hw/predecoder_kernel.xclbin`, gitignored). Whether to
  actually load it onto the real U55C card via `host/run_hw.py` /
  `host_predecoder.cpp` (exercising `run_on_hardware`, currently a
  `NotImplementedError` stub -- see the Phase 5 entries above) is a
  separate, not-yet-taken step.
