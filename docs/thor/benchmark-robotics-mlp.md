| Run | Engine / variant | Context | Status | Scope | Samples | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| dd616062-eab4-4907-81a4-cc607ff8b78a | synthetic-gemma300m-mlp-eager | 50 | PASS | robotics_block | 90 | 1.393712 | 1.451677 | 1.509884 |
| 8ec536d3-a2b7-4eb5-b86d-f68be98c0a81 | synthetic-gemma300m-mlp-graph | 50 | PASS | robotics_block | 90 | 0.695296 | 0.699107 | 0.705056 |

Same-cache decode replays are not autoregressive generation. Read each run's scope, accuracy gates, power profile and contention metadata before comparing. Tail estimates from small sample sets have limited precision.

These are synthetic 50-token, width-1024/MLP-4096 RMS/GELU/residual probes, ten repetitions with shared random weights. Median event intervals are 1.394 ms eager and 0.695 ms CUDA Graph; ninety checked samples each. Both have NRMSE 0.5808% versus the matched-rounding FP32-linear oracle (2% gate). Eager event intervals include host enqueue gaps. Fixed order, automatic clocks and uncontrolled external GPU load preclude an isolated speedup claim. No checkpoint, attention, adaptive RMS, action-head, flow-integration or vision-to-action coverage is implied. See [integration scope and operator gap](VLA_PORTING.md).
