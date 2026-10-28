| Run | Engine / variant | Context | Status | Scope | Samples | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| d4e56fe1-bf56-4bbe-94e4-70855e263320 | native:8k-256-compensated-stage-1-pages-9 | 1024 | PASS | decode | 90 | 10.977648 | 14.204891 | 14.468038 |
| 5998cd13-47b9-45aa-a100-4bba189fd9f6 | native:8k-256-compensated-stage-2-pages-17 | 1024 | PASS | decode | 90 | 11.431744 | 14.576434 | 14.889647 |
| 2eb9fb73-ea9a-46f8-be84-34ac21f2770d | native:8k-256-compensated-stage-3-pages-25 | 1024 | PASS | decode | 90 | 11.547760 | 14.731000 | 14.847174 |
| 19ce8878-1690-4882-b2a9-767be0567b60 | native:compensated-attention-stage-1-pages-5 | 1024 | PASS | decode | 90 | 10.964320 | 14.259559 | 14.497075 |
| c007599f-883f-4754-8541-cdd101968541 | native:compensated-attention-stage-2-pages-9 | 1024 | PASS | decode | 90 | 11.354656 | 14.629720 | 15.020260 |
| a991da43-32bb-43ef-8c95-7c8e4279d918 | native:compensated-attention-stage-3-pages-13 | 1024 | PASS | decode | 90 | 11.563568 | 14.672430 | 14.823763 |

Same-cache decode replays are not autoregressive generation. Read each run's scope, accuracy gates, power profile and contention metadata before comparing. Tail estimates from small sample sets have limited precision.
