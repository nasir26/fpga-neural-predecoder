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
