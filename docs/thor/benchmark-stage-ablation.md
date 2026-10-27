| Run | Engine / variant | Context | Status | Scope | Samples | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| 2e36d8e8-a9ea-4254-8cfe-b6cf22108ced | native:reference-rounding-stage-1-pages-5 | 1024 | PASS | decode | 90 | 11.664304 | 15.213587 | 15.788075 |
| c878d9ed-296b-4074-be83-d5f51aef09fa | native:reference-rounding-stage-2-pages-9 | 1024 | PASS | decode | 90 | 11.898128 | 15.220659 | 15.440652 |
| 13161e74-7bf4-4876-8e3e-8d46eb810652 | native:reference-rounding-stage-3-pages-13 | 1024 | PASS | decode | 90 | 11.984224 | 15.658181 | 15.999423 |
| e07053d9-7e7b-4db4-bfb2-053713f89b3d | native:reference-rounding-stage-1-pages-13 | 1024 | PASS | decode | 90 | 11.689488 | 15.136200 | 15.770778 |
| 8c558240-ba76-4547-9d18-5aacd2bbe54e | native:reference-rounding-stage-2-pages-13 | 1024 | PASS | decode | 90 | 11.990112 | 15.526478 | 15.708561 |

Same-cache decode replays are not autoregressive generation. Read each run's scope, accuracy gates, power profile and contention metadata before comparing. Tail estimates from small sample sets have limited precision.
