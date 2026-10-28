# Initial compensated-attention tuning sweep

Context 1024, batch 1, BF16, frozen-cache one-token replay. Existing GPU contention and clocks are uncontrolled. All 14 pilots passed ten timed correctness checks each. The three finalists passed 90 checks each. No default is selected. Nominal compiler register caps did not take effect; these are preserved negative results, not register-cap ablations. Source hash: `bc5210fef0090f09c7697b1cc6859d133e6babdf2eabb439f65f51fa6751f016`.

| Variant | Pilot p50 ms | Final p50 ms | Final p99 ms | Registers/thread | Local B/thread |
|---|---:|---:|---:|---:|---:|
| compensated-attention | 14.611 | — | — | 96 | 208 |
| consumer_warps-4 | 14.325 | 11.475 | 14.88 | 255 | 160 |
| consumer_warps-8 | 14.375 | — | — | 168 | 160 |
| static_registers-32 | 14.552 | — | — | 96 | 208 |
| static_registers-64 | 14.407 | — | — | 96 | 208 |
| static_registers-128 | 14.41 | — | — | 96 | 208 |
| workers-12 | 14.771 | — | — | 96 | 144 |
| workers-16 | 14.677 | — | — | 96 | 144 |
| backoff_ns-0 | 14.37 | 11.502 | 14.866 | 96 | 208 |
| backoff_ns-100 | 14.503 | — | — | 96 | 208 |
| backoff_ns-500 | 14.374 | 11.476 | 14.844 | 96 | 208 |
| scheduler-zz | 15.015 | — | — | 96 | 208 |
| scheduler-wave | 15.193 | — | — | 96 | 208 |
| scheduler-dag | 14.388 | — | — | 96 | 208 |

Observed fastest finalist key: `4f8b39a810e2f66f44e89b664fec6d81c5ae5bd7d8e3d4b93ee76e25bc536b86`. The observed p50/p99 frontier is retained in [manifest.json](sweep-compensated-initial/manifest.json). Finalist selection is biased by pilot ranking; sequential shared-load trials do not establish statistical significance or a causal speedup. Native temporal qualification remains open independently of these passing frozen-cache checks.
