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
