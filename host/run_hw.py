# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
"""XRT host driver for predecoder_kernel, via pyxrt.

Honest fallback, never silent: if no xclbin is available (currently always
true -- no `hw` bitstream has been built; see docs/DECISIONS.md), this
falls back to src/model/golden.py and labels every result "simulated": True.
The two code paths report the identical result schema so downstream
consumers can't accidentally treat a simulated result as a real one without
checking the "simulated" field, but the field is always present and always
printed loudly.

End-to-end latency accounting, per the plan: host-to-device transfer,
kernel, device-to-host transfer, and the residual PyMatching decode are all
reported separately, plus an honest total. Kernel/H2D/D2H are None in the
simulated path (there is no hardware to time); PyMatching decode time is
real in both paths.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
XRT_PYTHON_PATH = "/opt/xilinx/xrt/python"

from src.model.edges import apply_local_edge_corrections
from src.model.golden import extract_golden_layers, golden_forward
from src.model.qat import QATPreDecoder3D
from src.qec import metrics as qec_metrics
from src.qec.circuits import NoiseModel, build_rotated_surface_code_circuit, detector_coords_to_grid_index
from src.qec.dem import build_channel_observable_map, build_detector_error_model, build_matching_graph
from src.qec.sampling import diff_frames, sample_detection_volume

BEAT_BYTES = 512 // 8


def try_load_hardware(xclbin_path: Path):
    """Returns (device, kernel) if a real xclbin loaded successfully, else
    None. Never raises: any failure (no xclbin file, no pyxrt, no card,
    kernel not found in the xclbin) means "fall back", not "crash".
    """
    if not xclbin_path.exists():
        return None
    if XRT_PYTHON_PATH not in sys.path:
        sys.path.insert(0, XRT_PYTHON_PATH)
    try:
        import pyxrt
    except ImportError:
        return None
    try:
        device = pyxrt.device(0)
        xclbin_uuid = device.load_xclbin(str(xclbin_path))
        kernel = pyxrt.kernel(device, xclbin_uuid, "predecoder_kernel")
        return device, kernel
    except Exception:
        return None


def run_on_hardware(device, kernel, syndrome_beats: list, R: int) -> list:
    """Not exercised as of this commit: no xclbin exists to load, so
    try_load_hardware always returns None and this function is never
    called in practice. Written against the real pyxrt API (verified via
    host_predecoder.cpp compiling and linking against libxrt_coreutil), for
    when a working bitstream exists.
    """
    import pyxrt

    results = []
    syndrome_bank = kernel.group_id(0)
    correction_bank = kernel.group_id(1)
    for beat_bytes in syndrome_beats:
        t0 = time.perf_counter()
        syndrome_bo = pyxrt.bo(device, BEAT_BYTES, pyxrt.bo.flags.normal, syndrome_bank)
        correction_bo = pyxrt.bo(device, BEAT_BYTES, pyxrt.bo.flags.normal, correction_bank)
        syndrome_bo.write(beat_bytes, 0)
        syndrome_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, BEAT_BYTES, 0)
        t1 = time.perf_counter()

        run = kernel(syndrome_bo, correction_bo)
        run.wait()
        t2 = time.perf_counter()

        correction_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, BEAT_BYTES, 0)
        out_bytes = bytes(correction_bo.read(BEAT_BYTES, 0))
        t3 = time.perf_counter()

        results.append(
            {
                "h2d_us": (t1 - t0) * 1e6,
                "kernel_us": (t2 - t1) * 1e6,
                "d2h_us": (t3 - t2) * 1e6,
                "correction_beat": out_bytes,
            }
        )
    return results


def run_golden_fallback(model: QATPreDecoder3D, ckpt: dict, distance: int, p: float, shots: int, seed: int) -> dict:
    R = ckpt["R"]
    basis = ckpt["basis"]
    edge_offsets = ckpt["edge_offsets"]

    noise = NoiseModel.uniform_depolarizing(p)
    circuit = build_rotated_surface_code_circuit(distance, distance, basis, noise)
    dem = build_detector_error_model(circuit)
    matcher = build_matching_graph(dem)
    obs_map = build_channel_observable_map(circuit, edge_offsets)
    grid_index = detector_coords_to_grid_index(circuit)
    t_idx, row_idx, col_idx = grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]

    sampled = sample_detection_volume(circuit, distance, distance, shots, seed)
    diff = diff_frames(sampled.volume)

    t_model0 = time.perf_counter()
    with torch.no_grad():
        x = torch.from_numpy(diff).float().unsqueeze(1)
        logits = model(x)
        activation = (torch.sigmoid(logits) > 0.5).numpy().astype(np.uint8)
    correction = apply_local_edge_corrections(activation, edge_offsets)
    t_model1 = time.perf_counter()

    residual_volume = sampled.volume ^ correction
    residual_flat = residual_volume[:, t_idx, row_idx, col_idx].astype(bool)

    t_decode0 = time.perf_counter()
    residual_predicted = matcher.decode_batch(residual_flat).astype(bool)
    t_decode1 = time.perf_counter()

    flat_act = activation[:, :, t_idx, row_idx, col_idx].astype(np.uint32)
    local_obs_parity = ((flat_act.reshape(shots, -1) @ obs_map.flatten().astype(np.uint32)) % 2).astype(bool)
    predicted = residual_predicted ^ local_obs_parity.reshape(-1, 1)

    ler, ci_low, ci_high = qec_metrics.logical_error_rate(predicted, sampled.observables)
    model_us_per_shot = (t_model1 - t_model0) * 1e6 / shots
    decode_us_per_shot = (t_decode1 - t_decode0) * 1e6 / shots

    return {
        "simulated": True,
        "simulated_reason": "no xclbin available; ran src/model/golden.py-equivalent (QATPreDecoder3D) instead of real hardware",
        "distance": distance,
        "R": R,
        "p": p,
        "shots": shots,
        "seed": seed,
        "ler": ler,
        "ler_ci_low": ci_low,
        "ler_ci_high": ci_high,
        "mean_latency_us_per_shot": {
            "host_to_device": None,
            "kernel": None,
            "device_to_host": None,
            "predecoder_model": model_us_per_shot,
            "pymatching_decode": decode_us_per_shot,
            "total_honest": model_us_per_shot + decode_us_per_shot,
        },
        "mean_latency_us_per_round": {
            "host_to_device": None,
            "kernel": None,
            "device_to_host": None,
            "predecoder_model": model_us_per_shot / distance,
            "pymatching_decode": decode_us_per_shot / distance,
            "total_honest": (model_us_per_shot + decode_us_per_shot) / distance,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "models" / "predecoder_R5_int8.pt")
    parser.add_argument("--xclbin", type=Path, default=REPO_ROOT / "hls" / "predecoder_kernel.xclbin")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.003)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = QATPreDecoder3D(
        depth=ckpt["depth"], width=ckpt["width"], out_channels=ckpt["out_channels"], precision=ckpt["precision"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    hw = try_load_hardware(args.xclbin)
    if hw is None:
        print(
            f"WARNING: no usable xclbin at {args.xclbin} (or pyxrt/device unavailable). "
            f"Falling back to the golden model. RESULTS ARE SIMULATED, NOT FROM REAL HARDWARE.",
            file=sys.stderr,
        )
        result = run_golden_fallback(model, ckpt, args.distance, args.p, args.shots, args.seed)
    else:
        device, kernel = hw
        raise NotImplementedError(
            "a real xclbin was found, but the real-hardware decode/timing aggregation path "
            "(run_on_hardware -> LER/latency summary) is not implemented; only the fallback "
            "path has been exercised so far, see docs/LIMITATIONS.md"
        )

    out_dir = REPO_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"host_run_d{args.distance}_p{args.p}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path} (simulated={result['simulated']})")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
