// SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
// SPDX-License-Identifier: Apache-2.0
//
// Streaming 3D-CNN pre-decoder kernel, INT8, scoped to a fixed distance
// (R=5, T=H=W=6; see docs/DECISIONS.md 2026-08-17 "HLS kernel scoped to a
// fixed distance"). Matches src/model/golden.py's datapath exactly: two
// hidden Conv3d(k=3, same padding) + ReLU + UINT8 requant layers, a 1x1x1
// output conv producing NUM_CHANNELS raw accumulators, then the
// per-local-edge correction + observable-parity stage
// (src/model/edges.py's apply_local_edge_corrections, in hardware).
#pragma once

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "kernel_config.h"

using namespace predecoder_kernel_config;

typedef ap_uint<1> voxel_t;   // raw syndrome bit
typedef ap_uint<8> act_t;     // hidden-layer activation (unsigned, post-ReLU-requant)
typedef ap_int<8> weight_t;   // quantized weight (INT8, symmetric)
typedef ap_int<32> acc_t;     // conv accumulator, including integer bias
typedef ap_fixed<48, 8> mult_t;  // requantization multiplier (see docs/DECISIONS.md);
                                  // 8 integer bits (measured ratios top out ~2.03,
                                  // generous margin), 40 fractional bits

constexpr int LAYER0_OUT_C = 64;
constexpr int LAYER1_OUT_C = 64;
constexpr int LAYER2_OUT_C = NUM_CHANNELS;  // == 14

struct ActVec {
    act_t v[64];  // only the first IN_C entries are meaningful for a given layer's input
};
struct Logits {
    acc_t v[NUM_CHANNELS];
};

// syndrome_hbm / correction_hbm: one 512-bit beat each. syndrome_hbm's low
// T*H*W bits are the raster-order (t outer, y, x inner) input syndrome.
// correction_hbm's low T*H*W bits are the raster-order correction bits;
// bit T*H*W is the observable-parity bit for this shot; remaining bits 0.
void predecoder_kernel(const ap_uint<512>* syndrome_hbm, ap_uint<512>* correction_hbm);
