// SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
// SPDX-License-Identifier: Apache-2.0
//
// Sliding-window ("frame buffer") 3x3x3 window former for one 3D conv layer
// with "same" (zero) padding, matching src/model/golden.py's conv3d_int
// exactly: for output voxel (t,y,x), the window covers input
// (t-1..t+1, y-1..y+1, x-1..x+1), reading zero for any coordinate outside
// [0,T)x[0,H)x[0,W).
//
// Design: three full-frame buffers (each MAX_H x MAX_W x IN_C), used as a
// 3-deep ring indexed by (write_slot) rather than copied: at any time they
// hold frames t-2, t-1, and the frame currently being written (t). The two
// "history" frames (t-2, t-1) are complete, so their 3x3 spatial window
// around (y,x) is formed by direct (bounds-checked) indexing. The
// currently-arriving frame is not yet complete, so it is only ever read at
// the single (y,x) position just written (push_voxel keeps the running
// frame buffer for the *next* full-frame window, one frame-time later); the
// window position for the current voxel's dz=+1 (this frame) plane comes
// from that same running write, using the same bounds-checked read against
// what has been written so far being unnecessary, because that plane's
// content, once the whole frame is written, is exactly this frame's data --
// so we simply read it after the corresponding frame has fully advanced
// into the "t-1" or "t-2" slot. Concretely: push_voxel(t,y,x) writes into
// slot(t); get_window(t,y,x) reads slot(t-1)'s spatial window for dz=0,
// slot(t)'s for dz=+1 is not yet complete and not needed -- windows are only
// emitted for t one frame *behind* the write head (see PredecoderKernel).
#pragma once

#include <ap_int.h>

template <typename T, int IN_C, int MAX_H, int MAX_W>
class FrameBuffer3D {
public:
    FrameBuffer3D() { reset(); }

    void reset() {
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    RESET:
        for (int s = 0; s < 3; s++)
            for (int y = 0; y < MAX_H; y++)
                for (int x = 0; x < MAX_W; x++)
                    for (int c = 0; c < IN_C; c++) {
#pragma HLS PIPELINE II=1
                        buf[s][y][x][c] = 0;
                    }
    }

    // Write the voxel at real frame index t (>=0), row y, col x into its
    // ring slot. t must be non-decreasing across calls, starting at 0.
    void write_voxel(int t, int y, int x, const T pixel[IN_C]) {
#pragma HLS INLINE
        int slot = t % 3;
        for (int c = 0; c < IN_C; c++) {
#pragma HLS UNROLL
            buf[slot][y][x][c] = pixel[c];
        }
    }

    // Read the value at (frame index tt, row yy, col xx), returning zero if
    // outside [0,T)x[0,H)x[0,W) (the "same" padding). tt must be one of the
    // three most-recently-written frame indices (t, t-1, t-2 relative to
    // the last write_voxel call), i.e. still resident in the ring.
    void read(int tt, int yy, int xx, int T_real, int H, int W, T out[IN_C]) const {
#pragma HLS INLINE
        bool in_bounds = (tt >= 0) && (tt < T_real) && (yy >= 0) && (yy < H) && (xx >= 0) && (xx < W);
        int slot = ((tt % 3) + 3) % 3;
        for (int c = 0; c < IN_C; c++) {
#pragma HLS UNROLL
            out[c] = in_bounds ? buf[slot][yy][xx][c] : T(0);
        }
    }

private:
    T buf[3][MAX_H][MAX_W][IN_C];
};

// Forms the full 3x3x3xIN_C window for output voxel (t,y,x) from a
// FrameBuffer3D that has already had write_voxel called for every real
// voxel up to and including (t+1, H-1, W-1) reached in raster order (i.e.
// the caller streams input one frame ahead of the window it emits, so that
// frame t+1 -- needed for the window's dz=+1 plane -- is already buffered).
template <typename T, int IN_C, int MAX_H, int MAX_W>
void form_window(
    const FrameBuffer3D<T, IN_C, MAX_H, MAX_W>& fb,
    int t, int y, int x, int T_real, int H, int W,
    T window[3][3][3][IN_C]
) {
#pragma HLS INLINE
    for (int dz = 0; dz < 3; dz++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
#pragma HLS UNROLL
                T pix[IN_C];
                fb.read(t - 1 + dz, y - 1 + dy, x - 1 + dx, T_real, H, W, pix);
                for (int c = 0; c < IN_C; c++) {
#pragma HLS UNROLL
                    window[dz][dy][dx][c] = pix[c];
                }
            }
        }
    }
}
