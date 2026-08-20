# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""Generates committed test vectors (tests/vectors/) for the HLS testbench:
real sampled syndrome shots at d=5, their golden.py correction + observable
parity, in a simple text format tb_predecoder.cpp can parse without any
Python/JSON dependency.

Format (vectors_int8.txt): one line per shot:
  <216 syndrome bits><space><216 correction bits><space><obs_parity bit>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_coords_to_grid_index
from src.qec.dem import build_channel_observable_map
from src.qec.sampling import sample_detection_volume
from src.model.edges import apply_local_edge_corrections
from src.model.golden import extract_golden_layers, golden_forward
from src.model.qat import QATPreDecoder3D


def main(num_shots: int = 200, seed: int = 7) -> None:
    ckpt = torch.load(REPO_ROOT / "models" / "predecoder_R5_int8.pt", map_location="cpu", weights_only=True)
    model = QATPreDecoder3D(
        depth=ckpt["depth"], width=ckpt["width"], out_channels=ckpt["out_channels"], precision=ckpt["precision"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    R = ckpt["R"]
    basis = ckpt["basis"]
    edge_offsets = ckpt["edge_offsets"]
    layers = extract_golden_layers(model)

    circuit = build_rotated_surface_code_circuit(R, R, basis, NoiseModel.uniform_depolarizing(0.003))
    obs_map = build_channel_observable_map(circuit, edge_offsets)
    grid_index = detector_coords_to_grid_index(circuit)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]

    sampled = sample_detection_volume(circuit, R, R, num_shots, seed)
    T, H, W = sampled.volume.shape[1:]
    assert T == H == W == R + 1

    lines = []
    for s in range(num_shots):
        x_np = sampled.volume[s].astype(np.uint8)  # (T,H,W), 0/1
        acc = golden_forward(x_np, layers)  # (C,T,H,W) int64
        activation = (acc > 0).astype(np.uint8)[None, ...]  # (1,C,T,H,W)
        correction = apply_local_edge_corrections(activation, edge_offsets)[0]  # (T,H,W)

        # observable parity, matching bench/run_predecoder.py's local_correction_observable_parity
        flat_act = activation[0][:, t_idx, row_idx, col_idx]  # (C, num_detectors)
        parity = int((flat_act.astype(np.uint32) * obs_map.astype(np.uint32)).sum() % 2)

        syn_bits = "".join(str(int(b)) for b in x_np.flatten())
        corr_bits = "".join(str(int(b)) for b in correction.flatten())
        lines.append(f"{syn_bits} {corr_bits} {parity}")

    out_path = REPO_ROOT / "tests" / "vectors" / "vectors_int8.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}: {num_shots} shots, T=H=W={T}")


if __name__ == "__main__":
    main()
