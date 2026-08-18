// SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
// SPDX-License-Identifier: Apache-2.0
//
// XRT host driver for predecoder_kernel (see hls/predecoder_kernel.cpp).
// Not exercised against real hardware: no xclbin has been built for this
// kernel (the `hw` implementation build is gated behind BUILD_HW=1 and has
// not been run, since hls/predecoder_kernel.cpp's BRAM usage is currently
// well over the U55C's real budget; see docs/DECISIONS.md). This file is
// written to be correct against the real XRT C++ API and the kernel's
// actual two-argument (syndrome_hbm, correction_hbm) interface, but has
// only been compile-checked, not run. host/run_hw.py is the honest,
// actually-exercised path: it falls back to the golden model whenever no
// xclbin is available, which is the case right now.
//
// Design, per the plan's Phase 5 requirements:
//   - HBM bank assignment: syndrome_hbm and correction_hbm are placed in
//     the memory banks the kernel's own m_axi bundles (gmem0/gmem1) were
//     assigned to, queried via kernel::group_id(), not hardcoded, so this
//     stays correct if the bundle-to-bank mapping changes at synthesis time.
//   - Ping-pong buffers + ring buffer flow control: RING_DEPTH independent
//     (input bo, output bo, run) slots are cycled through, so shot N+1's
//     host-to-device transfer can overlap shot N's kernel execution and
//     shot N-1's device-to-host transfer, instead of fully serializing
//     every shot.
//   - Event-based timing: std::chrono timestamps bracket each stage
//     (host-to-device sync, kernel start/wait, device-to-host sync) per
//     shot, not just a single end-to-end wall clock, so the latency
//     breakdown is honest about where time actually goes.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

namespace {

constexpr size_t BEAT_BYTES = 512 / 8;  // one ap_uint<512> beat per shot
constexpr int RING_DEPTH = 4;

struct StageTiming {
    double h2d_us;
    double kernel_us;
    double d2h_us;
};

double us_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Slot {
    xrt::bo syndrome_bo;
    xrt::bo correction_bo;
    xrt::run run;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: host_predecoder <xclbin_path> <syndrome_beats_bin> <num_shots>\n");
        return 1;
    }
    const std::string xclbin_path = argv[1];
    const std::string shots_path = argv[2];
    const int num_shots = std::atoi(argv[3]);

    std::ifstream shots_file(shots_path, std::ios::binary);
    if (!shots_file) {
        std::fprintf(stderr, "could not open %s\n", shots_path.c_str());
        return 1;
    }
    std::vector<uint8_t> all_beats(static_cast<size_t>(num_shots) * BEAT_BYTES);
    shots_file.read(reinterpret_cast<char*>(all_beats.data()), all_beats.size());
    if (!shots_file) {
        std::fprintf(stderr, "shots file shorter than num_shots * %zu bytes\n", BEAT_BYTES);
        return 1;
    }

    xrt::device device(0);
    xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_path);
    xrt::kernel kernel(device, xclbin_uuid, "predecoder_kernel");

    // HBM bank assignment: ask the kernel which memory group each m_axi
    // argument was placed in at synthesis time, rather than assuming.
    const xrt::memory_group syndrome_bank = kernel.group_id(0);
    const xrt::memory_group correction_bank = kernel.group_id(1);

    std::vector<Slot> ring;
    ring.reserve(RING_DEPTH);
    for (int i = 0; i < RING_DEPTH; i++) {
        Slot slot;
        slot.syndrome_bo = xrt::bo(device, BEAT_BYTES, syndrome_bank);
        slot.correction_bo = xrt::bo(device, BEAT_BYTES, correction_bank);
        slot.run = xrt::run(kernel);
        slot.run.set_arg(0, slot.syndrome_bo);
        slot.run.set_arg(1, slot.correction_bo);
        ring.push_back(std::move(slot));
    }

    std::vector<StageTiming> timings;
    timings.reserve(num_shots);

    for (int shot = 0; shot < num_shots; shot++) {
        Slot& slot = ring[shot % RING_DEPTH];

        if (shot >= RING_DEPTH) {
            // Ring buffer flow control: this slot is reused, so wait for
            // its previous run to finish before overwriting its buffers.
            slot.run.wait();
        }

        auto t0 = std::chrono::steady_clock::now();
        auto* mapped_in = slot.syndrome_bo.map<uint8_t*>();
        std::memcpy(mapped_in, all_beats.data() + static_cast<size_t>(shot) * BEAT_BYTES, BEAT_BYTES);
        slot.syndrome_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t1 = std::chrono::steady_clock::now();

        slot.run.start();
        slot.run.wait();
        auto t2 = std::chrono::steady_clock::now();

        slot.correction_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto t3 = std::chrono::steady_clock::now();

        timings.push_back({us_between(t0, t1), us_between(t1, t2), us_between(t2, t3)});
    }

    // Drain any still-in-flight slots from the ring's tail.
    for (int i = 0; i < RING_DEPTH && i < num_shots; i++) {
        ring[i].run.wait();
    }

    double total_h2d = 0, total_kernel = 0, total_d2h = 0;
    for (const auto& t : timings) {
        total_h2d += t.h2d_us;
        total_kernel += t.kernel_us;
        total_d2h += t.d2h_us;
    }
    std::printf("shots=%d mean_h2d_us=%.2f mean_kernel_us=%.2f mean_d2h_us=%.2f\n", num_shots,
                total_h2d / num_shots, total_kernel / num_shots, total_d2h / num_shots);

    return 0;
}
