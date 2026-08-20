// SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
// SPDX-License-Identifier: Apache-2.0

#include "predecoder_kernel.h"
#include "line_buffer.h"
#include "weights_int8.h"

using namespace predecoder_weights;

// Wide product type for acc (up to ~2^27 in magnitude) * mult_t (ap_fixed<48,8>,
// up to ~2^8 integer range, 40 fractional bits): generously sized (not
// bit-width-optimal) to prioritize correctness for this first verified pass;
// see docs/DECISIONS.md.
typedef ap_fixed<96, 40> product_t;

static void unpack_input(const ap_uint<512>& beat, hls::stream<ActVec>& out) {
UNPACK:
    for (int t = 0; t < T; t++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                int idx = (t * H + y) * W + x;
                ActVec a;
                a.v[0] = beat[idx] ? act_t(1) : act_t(0);
                for (int c = 1; c < 64; c++) a.v[c] = 0;
                out.write(a);
            }
        }
    }
}

template <int IN_C>
static void conv3d_hidden_layer(
    hls::stream<ActVec>& in_stream,
    hls::stream<ActVec>& out_stream,
    const int8_t* weight_flat,  // flat [64][IN_C][3][3][3]
    const int32_t* bias,        // [64]
    const mult_t* requant_mult  // [64]
) {
    FrameBuffer3D<act_t, IN_C, H, W> fb;

CONV_T:
    for (int t = 0; t <= T; t++) {
    CONV_WRITE:
        if (t < T) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                    ActVec a = in_stream.read();
                    act_t pixel[IN_C];
                    for (int c = 0; c < IN_C; c++) {
#pragma HLS UNROLL
                        pixel[c] = a.v[c];
                    }
                    fb.write_voxel(t, y, x, pixel);
                }
            }
        }
        int emit_t = t - 1;
        if (emit_t < 0) continue;
    CONV_EMIT:
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                act_t window[3][3][3][IN_C];
                form_window(fb, emit_t, y, x, T, H, W, window);

                ActVec out;
            OC_LOOP:
                for (int oc = 0; oc < 64; oc++) {
                    // Deliberately NOT a single fully-unrolled dz*dy*dx*ic MAC:
                    // for IN_C=64 that is 1728 simultaneous reads into
                    // weight_flat every cycle, which forces the synthesizer to
                    // partition the weight array into ~1728 separately-banked
                    // slices -- the dominant driver of the 1159%-of-budget
                    // BRAM_18K overshoot recorded in results/metrics/hls_int8.json
                    // (see docs/DECISIONS.md). Unrolling only the 3x3x3 spatial
                    // taps (27-way) and pipelining the ic reduction serially
                    // cuts the parallel-port requirement on weight_flat by
                    // ~IN_C/1, at the cost of ~IN_C cycles per output channel
                    // instead of 1 -- latency was already far over target, so
                    // this trades a known-bad number for a worse one in
                    // exchange for actually fitting on the device.
                    acc_t partial[3][3][3];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=0
                PARTIAL_INIT:
                    for (int dz = 0; dz < 3; dz++)
                        for (int dy = 0; dy < 3; dy++)
                            for (int dx = 0; dx < 3; dx++) {
#pragma HLS UNROLL
                                partial[dz][dy][dx] = 0;
                            }

                IC_LOOP:
                    for (int ic = 0; ic < IN_C; ic++) {
#pragma HLS PIPELINE II=1
                    MAC:
                        for (int dz = 0; dz < 3; dz++)
                            for (int dy = 0; dy < 3; dy++)
                                for (int dx = 0; dx < 3; dx++) {
#pragma HLS UNROLL
                                    int widx = ((((oc * IN_C + ic) * 3 + dz) * 3 + dy) * 3 + dx);
                                    partial[dz][dy][dx] += (acc_t)weight_flat[widx] * (acc_t)window[dz][dy][dx][ic];
                                }
                    }

                    acc_t acc = bias[oc];
                SUM_PARTIAL:
                    for (int dz = 0; dz < 3; dz++)
                        for (int dy = 0; dy < 3; dy++)
                            for (int dx = 0; dx < 3; dx++) {
#pragma HLS UNROLL
                                acc += partial[dz][dy][dx];
                            }
                    if (acc < 0) acc = 0;  // ReLU

                    product_t scaled = product_t(acc) * product_t(requant_mult[oc]);
                    product_t rounded = scaled + product_t(0.5);  // acc >= 0 always here, so this is round-half-up
                    int q = (int)rounded;
                    if (q < 0) q = 0;
                    if (q > 255) q = 255;
                    out.v[oc] = (act_t)q;
                }
                out_stream.write(out);
            }
        }
    }
}

static void conv1x1_output_layer(hls::stream<ActVec>& in_stream, hls::stream<Logits>& out_stream) {
OUT_LOOP:
    for (int i = 0; i < T * H * W; i++) {
#pragma HLS PIPELINE II=1
        ActVec a = in_stream.read();
        Logits out;
    OC2_LOOP:
        for (int oc = 0; oc < NUM_CHANNELS; oc++) {
            acc_t acc = 0;
        MAC2:
            for (int ic = 0; ic < 64; ic++) {
#pragma HLS UNROLL
                int widx = oc * 64 + ic;
                acc += (acc_t)LAYER2_WEIGHT[widx] * (acc_t)a.v[ic];
            }
            acc += LAYER2_BIAS[oc];
            out.v[oc] = acc;
        }
        out_stream.write(out);
    }
}

static void edge_correction(hls::stream<Logits>& in_stream, ap_uint<512>& correction_beat) {
    // Deliberately NOT array-partitioned and NOT pipelined at II=1: each
    // voxel's edge channels can write to voxels reached by other iterations
    // too (the offset partner), a genuine cross-iteration read-modify-write
    // hazard on shared storage. Forcing II=1 here previously made synthesis
    // try to build full-width muxes for every data-dependent write across
    // overlapping in-flight iterations, which did not converge in
    // reasonable time (see docs/DECISIONS.md). This stage runs once per
    // shot and is not the throughput bottleneck (the conv layers are), so a
    // plain sequential loop over a normal BRAM-mapped array is the right
    // tradeoff here, not a resource-optimal one.
    ap_uint<1> correction[T][H][W];
    ap_uint<1> obs_parity = 0;

INIT:
    for (int t = 0; t < T; t++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                correction[t][y][x] = 0;
            }

EC_LOOP:
    for (int t = 0; t < T; t++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Logits lg = in_stream.read();
                int flat_idx = (t * H + y) * W + x;

                // channel 0: singleton, flips only (t,y,x)
                bool active0 = lg.v[0] > 0;
                if (active0) {
                    correction[t][y][x] ^= ap_uint<1>(1);
                    obs_parity ^= ap_uint<1>(OBS_PARITY_LUT[(0 * T + t) * H * W + y * W + x]);
                }

            EDGE_CH:
                for (int c = 1; c < NUM_CHANNELS; c++) {
                    bool active = lg.v[c] > 0;
                    if (!active) continue;
                    correction[t][y][x] ^= ap_uint<1>(1);
                    obs_parity ^= ap_uint<1>(OBS_PARITY_LUT[(c * T + t) * H * W + y * W + x]);

                    int dt = EDGE_OFFSETS[c - 1][0];
                    int dy = EDGE_OFFSETS[c - 1][1];
                    int dx = EDGE_OFFSETS[c - 1][2];
                    int tt = t + dt, yy = y + dy, xx = x + dx;
                    if (tt >= 0 && tt < T && yy >= 0 && yy < H && xx >= 0 && xx < W) {
                        correction[tt][yy][xx] ^= ap_uint<1>(1);
                        // The partner voxel is never itself an anchor for
                        // this specific fired edge (the anchor/offset
                        // convention is canonical, see src/qec/dem.py), so
                        // no separate obs-parity lookup for the partner: the
                        // observable-flip fact is recorded once, at the
                        // anchor, matching src/model/edges.py exactly.
                    }
                }
            }
        }
    }

PACK:
    ap_uint<512> beat = 0;
    for (int t = 0; t < T; t++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int idx = (t * H + y) * W + x;
                beat[idx] = correction[t][y][x];
            }
    beat[T * H * W] = obs_parity;
    correction_beat = beat;
}

void predecoder_kernel(const ap_uint<512>* syndrome_hbm, ap_uint<512>* correction_hbm) {
#pragma HLS INTERFACE m_axi port=syndrome_hbm bundle=gmem0 max_read_burst_length=16 num_read_outstanding=4 depth=1
#pragma HLS INTERFACE m_axi port=correction_hbm bundle=gmem1 max_write_burst_length=16 num_write_outstanding=4 depth=1
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    hls::stream<ActVec> s0("s0");
    hls::stream<ActVec> s1("s1");
    hls::stream<ActVec> s2("s2");
    hls::stream<Logits> s3("s3");
#pragma HLS STREAM variable=s0 depth=8
#pragma HLS STREAM variable=s1 depth=8
#pragma HLS STREAM variable=s2 depth=8
#pragma HLS STREAM variable=s3 depth=8

    ap_uint<512> in_beat = syndrome_hbm[0];
    ap_uint<512> out_beat;

    mult_t layer0_mult[64], layer1_mult[64];
CONVERT_MULT:
    for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II=1
        layer0_mult[i] = (mult_t)LAYER0_REQUANT_SCALE[i];
        layer1_mult[i] = (mult_t)LAYER1_REQUANT_SCALE[i];
    }

    unpack_input(in_beat, s0);
    conv3d_hidden_layer<1>(s0, s1, LAYER0_WEIGHT, LAYER0_BIAS, layer0_mult);
    conv3d_hidden_layer<64>(s1, s2, LAYER1_WEIGHT, LAYER1_BIAS, layer1_mult);
    conv1x1_output_layer(s2, s3);
    edge_correction(s3, out_beat);

    correction_hbm[0] = out_beat;
}
