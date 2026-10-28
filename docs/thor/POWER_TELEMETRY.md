# Scoped Thor power telemetry

The owned `tools.experiments.telemetry` collector records raw tegrastats lines with host monotonic receipt timestamps, converts current-reading mW/W units, and clips trapezoidal integration to a bracketed window. It refuses full-window energy when samples are missing/late, boundaries lack coverage, or fewer than ten samples/intervals support the duration. Host receipt is not a calibrated sensor timestamp. Missing GPU clocks remain unavailable. Only the collector’s own process group is stopped; no global tegrastats stop or power/clock changes are issued.

VIN is system consumption including module and carrier; VDD_GPU is the GPU rail, per [NVIDIA’s Thor power-monitor documentation](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html). Other workload consumption is included; no idle subtraction or causal device-energy attribution is made.

The bounded native command probe completes with 140 samples at a requested 100-ms interval. Estimated whole-command rail totals are 555.65 J VIN and 136.52 J VDD_GPU, over 14.22 s. This includes loading, prefill, reference work, warmup and checks, so both per-token energy fields are deliberately null. These are collector-validation results, not decode energy efficiency. The short-command test correctly leaves all full-window energy estimates unavailable; the help-only run records zero GPU rail consumption and must not be interpreted as inference. Raw JSON/JSONL and child benchmark output are preserved.

Camera coexistence has been removed from the active scope at the user’s direction; its interrupted exploratory study is archived under local storage, outside the tracked deliverable.
