# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
#
# Fast correctness-only check (csim, no csynth/cosim) for iterating on
# predecoder_kernel.cpp edits before paying for a full v++/csynth pass.
# Usage: vitis_hls -f hls/run_csim_only.tcl

open_project -reset predecoder_csim_check
set_top predecoder_kernel

add_files predecoder_kernel.cpp
add_files -tb tb_predecoder.cpp
add_files -tb ../tests/vectors/vectors_int8_small.txt

open_solution -reset "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default

csim_design

exit
