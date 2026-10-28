# Thor output completion diagnosis

Follow-up: all three implementation options have been tested end to end. See [actual implementation results](output-protocol-results/RESULTS.md).

The sampled QKV and up/gate storer bottleneck is the global readiness-publication region, rather than local output-slot signaling or TMA completion waiting. This narrows the previous output-slot lead; it does not establish a removable fraction of token latency.

Production source and its native binary were left unchanged. Observer-only changes extend the existing linked diagnostic worktree. No optimization was implemented.

## Measurement

Same batch-one, BF16 Llama 1B, context-1024, 100-token autoregressive workload as the prior diagnosis. Five warmup requests precede three measured requests. Timers sample consumer warp zero, loader lane zero and storer lane zero; copied buffers contain the final token at position 1123. The 3,028 matvec instructions per snapshot pass ownership and clock-partition checks.

Six uninstrumented requests bracketing the measurement have median 10.44796 ms/token. Three instrumented requests have median 10.54799 ms/token: **0.96% observer overhead**. Resources remain 96 registers, 10,384 static shared bytes, 400-byte local stack, and 4-byte spill loads/stores. These short measurements do not establish identical scheduling behavior.

| Operator | Consumer loop waiting for an output slot | Storer loop waiting for consumer arrivals | Storer loop inside store call | Global publication / store call | TMA completion wait / store call |
|---|---:|---:|---:|---:|---:|
| QKV | 57.07% | 36.37% | 62.52% | 69.06% | 2.26% |
| O projection | 4.93% | 92.30% | 5.41% | 0% | 14.56% |
| Up/gate | 69.96% | 16.40% | 81.65% | 83.79% | 0.91% |
| Down projection | 2.50% | 72.37% | 25.66% | 0% | 70.39% |
| Head | 1.88% | 73.41% | 25.36% | 0% | 76.83% |

Columns have different, explicitly named denominators. Store subphases are nested, and producer/consumer execution overlaps. Summing these percentages across roles or interpreting them as end-to-end speedup would be incorrect. Zero publication means no publication inside this particular store call, not no global barriers elsewhere in that operator.

QKV publication averages 3,844 local SM cycles per iteration; up/gate averages 4,998 cycles per iteration (about 9,996 per actual paired output because only odd iterations publish). Publication shares per snapshot are 68.91/69.58/68.70% for QKV and 83.90/83.87/83.61% for up/gate. Local slot-release signaling is only 0.28% and 0.79% of the respective storer loops.

The global publication timer encloses address/index construction and `gmem_add_release`. On Thor this helper uses a device-scope `cuda::atomic_ref::fetch_add` with acquire-release ordering. The observation attributes elapsed time to that region; it does **not** separate atomic contention, memory-ordering costs, instruction scheduling, or stalls left by preceding instructions. The remaining store-call span includes reduction, arithmetic, shared-memory conversion, store issuance, synchronization, and—for QKV—RoPE readiness. It is not a pure arithmetic measurement.

Control requests themselves produce two different token sequences; observer requests also vary. Initial reference checks pass, and instruction hashes match. This is the existing autoregressive workload, not a token-identical input replay. All three snapshots nevertheless agree on the dominant publication share. Numerics were not retuned.

## Interpretation and next experiment

1. QKV/up-gate are not primarily waiting for late consumer arrivals: their storer spends most of its loop processing completed outputs. O projection exhibits the opposite behavior. Treating all operators alike would miss this difference.
2. The large consumer output-slot wait is backpressure from the storer. The measured local semaphore signaling itself is small. Merely making that signaling faster is poorly motivated.
3. QKV/up-gate retain scratch slots while publishing global readiness. Releasing scratch after the TMA engine has finished reading it, before the global counter update, could allow more consumer overlap. It will not remove a sustained slow publication path; the storer can still become the throughput limit.
4. The strongest next bounded experiment is to isolate publication costs: compare equivalent ordering and contention patterns in a small probe, then consider per-producer completion records or safely batched publication. Preserve the visibility guarantees of all producers; do not simply replace acquire-release with relaxed atomics. Reduced publication granularity can delay downstream work, so measure end-to-end latency as well as local waits.

This concerns the persistent runtime's producer/consumer completion design. Evidence comes from Thor; no matching B200 experiment establishes that the cost is unique to Thor. These findings justify a focused protocol experiment, not a claimed performance gain or a broad rewrite. NVFP4/page changes remain a separate lead.

## Evidence and validation

Raw data and full observer patch: [output-waits](bottleneck/output-waits/). Generated aggregates: [summary.json](bottleneck/output-waits/summary.json). Reproduce aggregates with `python3 tools/analysis/report_output_waits.py docs/thor/bottleneck/output-waits`.

The first observer run is retained as `instrumented-initial-overlap.json` and excluded: a control initialization overlapped its end before being interrupted. The reported observer rerun and bracketing controls were serialized. An unrelated NVFP4 compilation was present in the workspace and was left untouched; shared system load and clocks were not controlled.

Memcheck and synccheck pass with zero errors; racecheck passes with zero hazards (zero errors and warnings). Sanitizer runs validate ownership but skip cycle-range arithmetic because instrumentation can overflow 32-bit local timers; performance snapshots pass those arithmetic checks without skipping.
