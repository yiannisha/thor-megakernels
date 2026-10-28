# Enforced register-cap tuning sweep

Context 1024, batch 1, BF16 frozen-cache one-token replay. Five passing pilots each contain ten timed correctness checks; three finalists each contain 90. The sixth configuration (128-register cap) fails occupancy preflight before execution. Existing GPU load and automatic clocks are uncontrolled. No default is selected. Static Thor kernels now use an explicit per-kernel register cap instead of launch bounds; dynamic Thor and H100/B200 retain launch bounds. Source hash: `f0091bc539433cc5e802ae270b3658ad4e22d3534539f3adc92fde1366077594`.

| Variant | Pilot p50 ms | Final p50 ms | Final p99 ms | Registers/thread | Local B/thread |
|---|---:|---:|---:|---:|---:|
| compensated-attention | 14.385 | 11.421 | 14.664 | 96 | 208 |
| consumer_warps-4 | 14.422 | 12.14 | 14.911 | 96 | 528 |
| consumer_warps-8 | 14.446 | 11.4 | 14.671 | 96 | 208 |
| static_registers-32 | 15.837 | — | — | 32 | 784 |
| static_registers-64 | 14.685 | — | — | 64 | 400 |
| static_registers-128 | REJECTED: zero resident CTAs | — | — | — | — |

The 32/64/96 caps are enforced in the compiled resource records, and the native preflight rejects an exceeded cap. Lower caps increase local storage in these builds; the 32-cap pilot has higher observed latency. The few finalist timing differences do not establish a controlled speedup. Pilot ranking introduces selection bias. See [complete raw manifest](sweep-enforced-registers/manifest.json); the earlier ineffective-cap sweep remains separately preserved. Passing frozen-cache checks do not close the independent temporal numerical qualification gate.

The 32-register full-model path also passes Compute Sanitizer memcheck with zero reported errors and one matched-cache timed numerical check ([raw result](register-cap32-model-memcheck.json), [sanitizer output](register-cap32-model-memcheck.txt)). Sanitized timings are not performance measurements. The isolated zero-iteration QKV attempt did not run because the synthetic fixture currently requires cap 96; its refusal is preserved in `register-cap32-qkv-zero-memcheck.txt`. Host validation passes 39 tests, with one opt-in CUDA test skipped by the unit runner; GPU experiments above ran separately.
