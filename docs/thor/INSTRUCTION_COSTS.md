# Native instruction timing and calibrated schedules

The optional timing build now copies timing slots collectively through the controller warp using generic global stores on Thor. This avoids the previous lane-0-only call into a helper containing a warp-wide barrier, and avoids asynchronous-proxy publication of timing records produced by other warps. The controller waits for every execution thread's instruction-completion arrival before copying or resetting a slot. A warp synchronization publishes its own final timestamp before distributed copying. Generic diagnostic events remain atomic; only unique role start/end slots are interpreted as durations.

The stable role-pair schema is loader 5/6, launcher 7/8, storer 9/10, and consumer warp `w` at `11+2*w` / `12+2*w`. The attention reduction input-ready diagnostic moved to 80..95 so it cannot overwrite consumer pairs. Zero endpoints are treated as missing. The exporter subtracts each pair modulo 2^32 and rejects ambiguous/reversed intervals of at least 2^31 cycles. Both endpoints come from the same warp lane; no timestamps are compared across warps or workers. Units are SM cycles, without assuming a fixed GPU clock.

`tools/experiments/instruction_profile.py` exports real Llama instruction words, worker/queue position, inclusive per-role cycles, maximum consumer span and original analytical cost. Spans include semaphore waits, memory stalls and collectives. They are not pure arithmetic costs or a global trace. The raw profile has numerical checks before and during collection, source/binary provenance, and the exporter source hash.

`Schedule.cost_model` is an optional callback used by both static DAG and wave schedulers. Its results must be finite and positive. Existing analytical estimates remain the default. `tools/experiments/calibrated_schedule.py` fits median inclusive maximum-consumer cycles by opcode and analytical shape cost. It requires matching model/weights, context, precision, numerical/resource configuration and native source. It rejects absent shape buckets rather than silently mixing units or extrapolating. A separate ready-only static list scheduler uses measured remaining critical-path length; the optional locality tie-break prefers workers that produced dependencies when estimated completion times tie. Queue coverage and dependency/order acyclicity are validated before tensorization. This is static queue placement, not a device-side ready queue or activation retention in registers/SMEM.

The first attempted calibrated run was rejected because the model adapter constructed queues with `globs.pos_id=0`, before installing the decode context. The actual profile correctly measured context 1024, exposing the mismatch through its attention cost bucket. The adapter now sets the context before constructing schedules. Earlier RR placements are unaffected; earlier cost-based scheduling datasets remain historical measurements with that analytical context error. Corrected profiles and controls are recorded under `instruction-profile-context-corrected/` and `calibrated-scheduling-context-corrected/`. The initial profile/overhead data and rejected attempt remain in `instruction-profile/` and `calibrated-scheduling/`.

Reproduction uses `source tools/thor/env.sh`, `configs/thor/cost-profile.json` for the timing build, and `configs/thor/compensated-attention.json` for the control. Build each through the existing sweep/build definitions; exact compiler arguments are retained in process records. Set `THOR_BINARY_DIR` and `THOR_CONFIG` to the corresponding build/config, then run:

```bash
python -m tools.experiments.instruction_profile --context 1024 --samples 3 --output local/instruction-profile.json
```

For a measured-cost static schedule, use an untimed build, a config with scheduler `dag` or `wave`, and select the raw profile and optional queue policy:

```bash
export THOR_COST_PROFILE="$PWD/local/instruction-profile.json"
export THOR_COST_POLICY=critical # greedy, critical, or locality
python -m tools.thor.bench --adapter tools.experiments.calibrated_schedule:make_adapter --warmups 5 --repeats 30 --batches 3 --output local/calibrated-benchmark.json
```

Instrumentation overhead is measured separately with matched control/timing builds, alternating order over three independently loaded batches. Scheduler comparison shuffles seven variants per batch with recorded seed 20261027. The shared GPU has unrelated workloads; comparisons are observational and cannot establish isolated causal speedups. Measured costs do not waive accumulated-cache numerical failures. No configuration is promoted by these experiments.

Corrected-source timing qualification passes memcheck, racecheck and synccheck with zero errors/hazards. Each of the three contexts produces 9,468 instruction observations (3,156 model instructions × three samples), with all numerical checks passing. These profiles cover the six opcodes present in this skip-attention-reduction Llama DAG; they do not profile unused opcodes or every attention partition shape.

| Build | Samples | Replay p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| control | 90 | 14.462 | 14.769 | 14.851 |
| timing | 90 | 14.469 | 14.894 | 14.996 |

Observed pooled p50 changes by +0.05% with timing enabled. This alternating-order shared-load result measures the instrumentation variant as a whole; it is not an isolated per-event cost. Both builds use the same cap/resources and mathematical configuration. All 180 timed correctness checks pass.

| Context | QKV | Attention | O projection | Up/gate | Down projection | LM head |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 131,092 | 16,326 | 9,168 | 487,545 | 215,624 | 3,822,056 |
| 1024 | 131,716 | 65,374 | 9,348 | 490,656 | 213,602 | 3,934,950 |
| 8192 | 132,338 | 454,356 | 9,218 | 492,744 | 219,954 | 3,682,477 |

The table reports median inclusive maximum-consumer cycles pooled by opcode, mixing the instruction sizes actually scheduled. Shape-specific calibration uses eleven opcode/analytical-cost buckets instead. In particular the LM head instruction spans hundreds of output tiles and is not comparable to a single O-projection tile. These spans cannot be summed across workers to obtain kernel latency.

The seven-policy context-1024 comparison completes all 630 timed correctness checks. Pooled medians range from 14.417 to 14.472 ms; no convincing measured-cost improvement is established. See [table, CSV and CDF](benchmark-calibrated-scheduling.md). Critical-path and producer-affinity policies produce exactly the same serialized queues on this DAG, so the latter supplies no activation-placement improvement here. All three distinct new queues pass memcheck, synccheck and racecheck (nine commands, zero errors/hazards) in `calibrated-scheduling-sanitizers/`. The affinity policy needs no separate duplicate sanitizer run because its serialized queue tensor, loaded binary, weights, context and numerical configuration match the critical policy. Fused activation retention and a model-level device ready queue remain separate work.

Final verification: the host suite passes 53 tests, with one optional GPU test skipped. H100/B200 compile checks and the primary Thor build succeed from the corrected source (`instruction-profile-context-corrected/compat-builds.json`); data-center hardware execution remains untested. The refreshed primary Thor binary completes the independent-native-KV 100-step math-reference check but **fails** steps 10/11/21: worst logit NRMSE 3.0851569%, hidden NRMSE 2.5564594%, all 100 greedy tokens agree (`temporal-baseline-current.json` in the same directory). This reproduces the earlier baseline failure on the current binary. No tolerance was relaxed.
