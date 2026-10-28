# Dynamic register allocation sweep

BF16, context 1024, batch 1, frozen-cache one-token replay. Six pilots pass ten timed correctness checks each; three finalists pass 90 each. The 128-register consumer target is rejected by the configuration constraint. Shared GPU contention and automatic clocks are uncontrolled. No default is selected. Source hash: `37d797985d3f24085793761c763c94f63fc15f5ac6496c26100ff6d349252958`.

| Variant | Pilot p50 ms | Final p50 ms | Final p99 ms | Initial registers/thread | Initial CTA pool | Role target sum |
|---|---:|---:|---:|---:|---:|---:|
| compensated-attention | 14.551 | 11.466 | 14.74 | 96 | 61440 | 61440 |
| register_mode-dynamic | 14.757 | 11.823 | 15.019 | 96 | 61440 | 61440 |
| consumer_registers-64 | 14.798 | 11.833 | 15.648 | 64 | 40960 | 40960 |
| consumer_registers-96 | 14.897 | — | — | 96 | 61440 | 57344 |
| consumer_registers-128 | INVALID | — | — | — | — | — |
| service_registers-32 | 16.073 | — | — | 96 | 61440 | 57344 |
| service_registers-48 | 15.311 | — | — | 96 | 61440 | 59392 |

Dynamic Thor kernels establish a known 24-register allocation before constructing VM state, synchronize all CTA threads, then increase each aligned warpgroup to its consumer/service target. All targets must fit the initial CTA pool; free registers elsewhere on the SM are not available to this CTA. Static Thor and data-center register paths retain their respective behavior. The static row reports the configured dynamic-role sum for context only; those role targets are inactive in static mode.

The observed dynamic finalists are slower than the static control in this run. This is an exploratory comparison with pilot selection bias, not a controlled causal result. Raw build commands, resource records, per-sample checks and p50/p99 frontier are in [the manifest](sweep-dynamic-bootstrap/manifest.json). Temporal model qualification remains open independently.

Admission negative test: consumer 104/service 72 satisfies the per-SM Python budget constraint, but its 62,464-register role total exceeds the compiled 61,440-register initial CTA pool. Native preflight rejects it before launch. [Raw rejection manifest](dynamic-pool-rejection/manifest.json) and [worker result](dynamic-pool-rejection/pilot.json) retain the failure.

Sanitizer validation: dynamic-control QKV zero-iteration/reuse passes racecheck and synccheck; negative-score attention passes racecheck. The service-32 full-model path passes memcheck. Unfiltered full-model synccheck reports 227 internal sanitizer launch errors on cuBLAS `nvjet` kernels and is **inconclusive**, despite application numerical completion. A separately preserved synccheck run filtered to native `megakernel` kernels passes with zero errors; this establishes native scope only. Commands, exact configurations and return codes are in [sanitizer records](dynamic-bootstrap-sanitizers.json). Sanitizer runs are not latency evidence. The host suite passes 39 tests with one opt-in CUDA skip.
