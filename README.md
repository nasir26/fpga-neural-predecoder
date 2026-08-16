# fpga-neural-predecoder

FPGA reimplementation, in progress, of the neural pre-decoder concept for
surface-code quantum error correction described by NVIDIA's `Ising-Decoding`
work (see `NOTICE`). This is an independent implementation with no NVIDIA
source code; the FPGA microarchitecture, quantization study, and latency
model are original contributions. Author: Nasir Ali, C-DAC Noida.

**Status: Phase 0 (environment and sanity) and Phase 1 (QEC data layer)
complete. Phase 2 not started.** This README will gain a method summary,
results table, and reproduce instructions once there are results to report;
see `claude_code_prompt_ising_fpga.md` in the repository root for the full
phased plan this project follows, and `docs/LIMITATIONS.md` /
`docs/DECISIONS.md` for scope reductions made along the way (notably: X/Z
basis only, no "four orientations"; uniform-depolarizing noise only, the
25-parameter noise model is deferred).

## Environment

- venv: Python 3.8.10 (see `docs/DECISIONS.md` for why, not 3.11)
- Dependencies: pinned in `requirements.txt`; exact resolved set in
  `docs/env.lock`
- `stim` is pinned to 1.12.1, not the latest release; see
  `docs/DECISIONS.md`
- `torch` is the CPU-only wheel; `torch.cuda.is_available()` is asserted
  False at every environment check

## Setup

```
make venv
make smoke
```

`make smoke` builds a `d=3, rounds=3` rotated surface-code circuit, samples
1000 shots, decodes with PyMatching, and prints the logical error rate, as a
sanity check that the toolchain works end to end.

## Hardware

Target: Xilinx Alveo U55C. This development host happens to have one
installed with a working XRT stack; see `docs/DECISIONS.md` for the full
inventory. Later phases will state plainly whether a given result came from
C simulation, cosimulation, or an actual on-card `hw` run.

## Limitations

Not yet written; will be added starting in Phase 1 and expanded through the
project (`docs/LIMITATIONS.md`).
