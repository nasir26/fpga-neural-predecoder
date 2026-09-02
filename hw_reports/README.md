# Archived synthesis and implementation reports

These are the raw Vivado/Vitis reports that the numbers in
`results/metrics/hw_build_int8.json` and the paper were extracted from. They
are archived here because `hls/_x/` is a build directory and is not tracked,
so without this copy the provenance of those numbers would be lost as soon as
the build tree was cleaned.

| File | What it backs |
|---|---|
| `predecoder_kernel_csynth.rpt` | Vitis HLS C-synthesis estimates (latency, II, per-layer resources) |
| `impl_1_kernel_util_routed.rpt` | Post-route utilisation scoped to `predecoder_kernel_1`, excluding the platform shell |
| `impl_1_hw_bb_locked_timing_summary_routed.rpt.gz` | Post-route timing summary; the WNS/TNS/WHS figures and the timing-closure claim (gzipped: 16 MB raw, 0.4 MB compressed) |

Decompress the timing summary with:

    gunzip -k impl_1_hw_bb_locked_timing_summary_routed.rpt.gz

The corresponding bitstream (`predecoder_kernel.xclbin`, 50 MB) is attached to
the GitHub Release rather than committed, since it is a large binary.
