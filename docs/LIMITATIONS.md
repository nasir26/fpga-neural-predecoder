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
