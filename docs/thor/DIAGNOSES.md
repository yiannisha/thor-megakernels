# Frozen-port diagnoses: traffic, TMA concurrency and VM waits

No production runtime or arithmetic changes were made. The observer exists only in a separate worktree, with its exact patch/build fingerprint preserved. These findings select possible follow-up experiments; they are not optimization speedups.

## 1. Source-level traffic accounting

The unchanged execution plan has 3,156 tasks. Coverage checks confirm that every linear-weight tile appears exactly once, including all four down-projection column ranges and the complete output head.

| Counted payload | Bytes per decode |
|---|---:|
| linear_weight_reads | 2,471,493,632 |
| activation_reads | 12,402,688 |
| normalization_weight_reads | 2,703,360 |
| rope_table_reads | 163,840 |
| qkv_writes | 98,304 |
| kv_reads_requested_with_tail_padding | 34,078,720 |
| attention_query_reads | 65,536 |
| attention_output_writes | 65,536 |
| residual_RMW_reads | 327,680 |
| residual_RMW_writes | 327,680 |
| silu_output_writes | 262,144 |
| logit_writes | 256,512 |
| instruction_fetch_reads | 404,480 |

Total counted payload is 2,522,650,112 bytes. Linear weights are **97.97%**. Repeated activation inputs are 11,939,840 bytes and repeated normalization weights are 2,568,192 bytes; together these are 0.575% of counted payload.

These are requested payload bytes, not DRAM transactions. Repeated small inputs may hit cache. Poll counts, cache-line amplification, spills and atomic transaction effects are not counted; their latency can matter despite small payloads. Atomic completion RMW operations and outside-VM payload estimates are recorded separately in the raw JSON. Thus this accounting supports deprioritizing bulk activation-traffic elimination, not declaring all synchronization negligible.

## 2. Matching TMA transfer probe

The probe uses native 16×512 BF16 TMA tiles, four chunks per 64-KiB iteration, twenty CTAs, 640 threads/CTA, sixteen consumer warps and a fixed thirteen-page allocation. One-, two- and three-stage cases change prefetch depth without changing allocation/residency. Issuer cases use one/two/four lanes in the loader warp. Model arithmetic and inter-task dependencies are absent; registers are not forced to match the full VM. All cases admit one CTA/SM.

Three independently initialized batches use shuffled case orders (seeds 71/72/73), five warmups and thirty timed launches per case. Consumer-read mode reads the complete shared payload and checks its final checksum against an independent global reader after each thirty-sample case. Discard mode waits for TMA completion without shared-data consumption. Tiny-output checksum fixtures exercise ring phases under all three sanitizers.

| Stages | Issuing lanes | Consume: median of batch medians ms | Batch range ms | Discard ms |
|---:|---:|---:|---:|---:|
| 3 | 1 | 9.5107 | 9.4961–9.5648 | 9.5028 |
| 1 | 1 | 9.3026 | 9.2937–9.3133 | 9.3175 |
| 2 | 1 | 9.5253 | 9.4564–9.5427 | 9.4990 |
| 3 | 2 | 9.5458 | 9.4806–9.5522 | 9.5271 |
| 3 | 4 | 9.4946 | 9.4610–9.5377 | 9.5537 |

The matching three-stage/single-issuer path reaches about 260 GB/s in this arithmetic-free probe. More issuing lanes do not materially improve it. One stage is slightly faster in the microbenchmark, but it has no matvec computation to overlap, so this does not establish a one-stage VM improvement. The probe demonstrates that one resident CTA/SM and one issuing lane can supply high bandwidth; a blanket shortage of loader concurrency is not established.

## 3. Scoped VM observer

A control–instrumented–control sequence uses five full 100-step warmup requests and three measured 100-step requests per phase. Control median is **10.3948 ms/token**, instrumented median **10.3929 ms/token** (-0.019%). This shows no measurable slowdown in this short check, not proof that clocks are free. Compiled register count, static SMEM, local storage and reported spill sizes remain unchanged.

Cycle snapshots are from the final position 1123 of each measured request. Only consumer warp 0 and loader lane 0 are sampled; 3,028 matvec instructions per snapshot are covered. Attention compute and other consumer warps are not sampled. All comparisons use elapsed clocks on the same SM; absolute clocks are not compared across SMs. Wait regions include primitive/timestamp costs and are not exact hardware stall counters.

| Family | Weight wait % | Output-slot wait % | Matvec work % | Other loop % |
|---|---:|---:|---:|---:|
| QKV | 9.74 | 56.14 | 28.60 | 5.52 |
| O projection | 39.04 | 5.56 | 41.38 | 14.02 |
| Up/gate | 2.74 | 69.33 | 24.37 | 3.56 |
| Down projection | 62.35 | 2.34 | 30.58 | 4.73 |
| Output head | 69.14 | 1.84 | 25.71 | 3.32 |

Pooled sampled consumer-loop cycles: weight readiness 31.08%, output-slot availability 37.94%, matvec work 26.75%. Separately, global dependency waits account for 13.35% of sampled input-setup-plus-loop cycles.

**These percentages are not fractions of end-to-end latency and cannot be treated as recoverable time.** Loaders, storers and other workers progress while the sampled consumer waits. An output-slot wait may reflect storer reduction/publication or late consumer arrivals; the observer does not distinguish those causes. This is evidence of family-dependent pipeline backpressure, not a universal weight-prefetch shortage.

Memcheck and synccheck pass for the observer. Initial racecheck reports zero hazards but the strict timing-value assertion fails because sanitizer-slowed output-head execution exceeds the 32-bit clock representation/range. That failed invocation is preserved. A second racecheck checks row ownership/bounds while skipping clock arithmetic, and passes with zero hazards; its twenty clock-range rows are output-head rows. Performance-cycle invariants are checked on every non-sanitized snapshot. Sanitizer timings are excluded from performance interpretation.

## Decision for follow-up

- Large reductions from eliminating repeated activations are not supported by the counted traffic volume.
- More weight-issuing lanes are not supported by the TMA probe.
- The output-slot/storer completion path is the strongest new runtime lead, particularly for paired-output tasks. First distinguish storer work/publication from late-arriving consumers before selecting a change.
- Weight waits remain prominent for long head/down-projection tasks; any output-path change must preserve their prefetch overlap.
- Hardware traffic/stall counters remain permission-blocked, so actual DRAM saturation versus internal stalls is not fully resolved.

## Reproduction and evidence

`traffic_accounting.py` runs against the unchanged native adapter and asserts exact instruction coverage. `tma_stream.cu` uses the pinned TK overlay; compile with C++20, `-DKITTENS_HOPPER`, native SM110a codegen, relaxed constexpr and extended lambda flags, the overlay include and `-lcuda`. Seeds are command-line arguments. Sanitizer builds use `TMA_DIAG_TILES=160`, `TMA_DIAG_WARMUPS=0`, `TMA_DIAG_REPEATS=1`; their timing is not a performance result.

The observer is generated by `tools/analysis/instrument_waits.py` in an isolated linked worktree. `waits/observer-source.json` preserves the exact patch against `87fd234`; `waits/native-build.json` and `waits/build.txt` identify the actual binary/resources. The worktree reuses the original local dependencies/model and pinned TK headers. Production source and binary remain unchanged. `collect_waits.py` is identical in both checkouts, with `THOR_BINARY_DIR` selecting the diagnostic binary only in its worktree. `waits/processes.json` records the sequential collection commands.

Raw traffic, all three shuffled TMA cohorts, sanitizer output, three VM snapshots, cycle schema and CSV summaries are under [bottleneck/](bottleneck/). Clock columns: 0 weight wait, 1 output-slot wait, 2 matvec work, 3 consumer-loop span, 4 loader reuse wait, 5 loader page wait, 6 loader-loop span, 7 global dependency wait, 8 consumer input-setup span, 9 unused, 10 opcode, 11 matvec iterations. The clock fields are per-instruction local-cycle durations; some are nested or overlap across roles.
