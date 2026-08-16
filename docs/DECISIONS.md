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
