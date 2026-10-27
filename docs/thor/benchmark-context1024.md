| Run | Engine / variant | Context | Status | Scope | Samples | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| 999dc53e-a300-4fd1-a1da-beaabb0d5a2e | native:upstream-like | 1024 | PASS | decode | 90 | 11.608128 | 15.020674 | 15.264781 |
| 304c2f78-bc91-49b7-bad1-6ca463be5ef5 | torch:upstream-like | 1024 | PASS | decode | 90 | 14.127152 | 17.524294 | 17.889095 |
| 1b3ca18e-51e5-4cfd-8ea5-e82c47a5aa40 | torch_graph:upstream-like | 1024 | PASS | decode | 90 | 12.800416 | 16.476936 | 17.770450 |
| a1817ff2-39a3-4504-bf84-201ef17e0584 | pyvm:upstream-like | 1024 | PASS | decode | 90 | 2240.013794 | 2268.319080 | 2277.470708 |

Same-cache decode replays are not autoregressive generation. Read each run's scope, accuracy gates, power profile and contention metadata before comparing. Tail estimates from small sample sets have limited precision.
