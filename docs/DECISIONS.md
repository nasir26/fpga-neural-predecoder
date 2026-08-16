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
