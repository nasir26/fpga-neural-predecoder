# SPDX-FileCopyrightText: Copyright (c) 2026 Nasir Ali, C-DAC Noida
# SPDX-License-Identifier: Apache-2.0
#
# csim, csynth, cosim always run. The full `hw` bitstream build (export_design
# -flow impl) is gated behind BUILD_HW=1 in the environment; do not set it
# without being asked (it can take hours). Usage:
#   vitis_hls -f hls/run_hls.tcl
#   BUILD_HW=1 vitis_hls -f hls/run_hls.tcl

open_project -reset predecoder_proj
set_top predecoder_kernel

add_files predecoder_kernel.cpp
add_files -tb tb_predecoder.cpp
add_files -tb ../tests/vectors/vectors_int8_small.txt

open_solution -reset "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default

csim_design

csynth_design

cosim_design

if {[info exists ::env(BUILD_HW)] && $::env(BUILD_HW) == "1"} {
    export_design -flow impl -rtl verilog
}

exit
