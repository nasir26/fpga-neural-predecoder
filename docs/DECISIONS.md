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
