# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
import hashlib

import numpy as np
import torch

from src.model.export import export_hls_header
from src.model.golden import extract_golden_layers
from src.model.qat import QATPreDecoder3D


def _make_layers(precision="int8"):
    torch.manual_seed(0)
    qat = QATPreDecoder3D(depth=2, width=4, out_channels=3, precision=precision)
    qat.train()
    with torch.no_grad():
        qat(torch.randint(0, 2, (4, 1, 6, 6, 6)).float())
    qat.eval()
    return extract_golden_layers(qat)


def test_export_writes_header_and_bin(tmp_path):
    layers = _make_layers()
    meta = export_hls_header(layers, precision="int8", R=5, output_dir=tmp_path)

    header_path = tmp_path / "weights_int8.h"
    bin_path = tmp_path / "weights_int8.bin"
    assert header_path.exists()
    assert bin_path.exists()

    header_text = header_path.read_text()
    assert meta["sha256"] in header_text
    assert "NUM_LAYERS = 3" in header_text
    assert "LAYER0_WEIGHT" in header_text


def test_export_sha256_matches_bin_contents(tmp_path):
    layers = _make_layers()
    meta = export_hls_header(layers, precision="int8", R=5, output_dir=tmp_path)
    bin_bytes = (tmp_path / "weights_int8.bin").read_bytes()
    assert hashlib.sha256(bin_bytes).hexdigest() == meta["sha256"]
    assert len(bin_bytes) == meta["weight_bytes"]


def test_export_is_deterministic(tmp_path):
    layers = _make_layers()
    meta1 = export_hls_header(layers, precision="int8", R=5, output_dir=tmp_path / "a")
    meta2 = export_hls_header(layers, precision="int8", R=5, output_dir=tmp_path / "b")
    assert meta1["sha256"] == meta2["sha256"]


def test_bin_roundtrip_reconstructs_exact_weights_and_biases(tmp_path):
    layers = _make_layers()
    meta = export_hls_header(layers, precision="int8", R=5, output_dir=tmp_path)
    blob = (tmp_path / "weights_int8.bin").read_bytes()

    for sec, layer in zip(meta["layers"], layers):
        w_bytes = blob[sec["weight_offset"] : sec["weight_offset"] + sec["weight_length"]]
        w_reconstructed = np.frombuffer(w_bytes, dtype=np.int8).reshape(sec["weight_shape"])
        np.testing.assert_array_equal(w_reconstructed, layer.weight_int.astype(np.int8))

        b_bytes = blob[sec["bias_offset"] : sec["bias_offset"] + sec["bias_length"] * 4]
        b_reconstructed = np.frombuffer(b_bytes, dtype=np.int32)
        np.testing.assert_array_equal(b_reconstructed, layer.bias_int.astype(np.int32))


def test_different_precisions_give_different_sha256(tmp_path):
    layers_int8 = _make_layers("int8")
    layers_ternary = _make_layers("ternary")
    meta8 = export_hls_header(layers_int8, precision="int8", R=5, output_dir=tmp_path / "int8")
    metat = export_hls_header(layers_ternary, precision="ternary", R=5, output_dir=tmp_path / "ternary")
    assert meta8["sha256"] != metat["sha256"]
