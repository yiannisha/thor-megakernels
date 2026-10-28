# Actual output protocol implementation results

The tested protocol changes offer small end-to-end differences on Thor with BF16 Llama 1B. Release-only readiness counters are the simplest candidate: confirmation median **10.35686 vs 10.39468 ms/token**, a **0.36% reduction (37.8 microseconds)**. The corrected asynchronous publication queue regresses median latency by **0.50%**, and all three comparison blocks are slower. No implementation produces a multi-percent median improvement.

The production checkout and its original binary remain unchanged. Actual implementation commits are preserved on experimental branches; this is a completed implementation/measurement exercise, not a microbenchmark proposal.

## Implementations

- **Early slot release:** QKV and paired up/gate release shared output slots after their TMA stores complete, before readiness publication. Arithmetic and scheduling are unchanged.
- **Release-only counters:** device-scope acquire-release `fetch_add` becomes release `fetch_add`; consumers retain acquire loads. Release RMWs on the same counter participate in prior producers' release sequences, preserving publication of all completed producers' stores to the final acquire observer.
- **Batched counters:** a per-instruction accumulator combines consecutive updates to the same readiness counter, flushing on counter change and instruction completion. It retains acquire-release global atomics and the existing downstream target counts. Static update counts fall from 3,072 to 1,024 for QKV and from 8,192 to 1,280 for up/gate: **79.55% fewer targeted RMW calls**. These are counts derived from the unchanged instruction schedule, not physical atomic transaction counters. Delayed publication can offset the benefit.
- **Decoupled publication:** a per-instruction eight-entry shared-memory queue transfers completed-output records from the storer to the existing launcher warp. The storer releases output scratch early; the launcher publishes readiness; the instruction cannot retire until publication is drained. The final implementation uses per-slot ready/consumed hardware mbarriers. It uses existing scratch and semaphore storage, adds no threads, and retains the original residency.

The shared helper is a runtime completion protocol; QKV/up-gate call sites integrate it because those are the store paths implicated by the diagnosis. No model math, model weights, quantization, attention implementation, or static task schedule was changed.

## First comparison: implemented variants against all options disabled

Three randomized blocks, ten full requests per variant per block. Values are medians of all 30 request-average token latencies. Percent reductions use this phase's own baseline.

| Implementation | Median ms/token | Median latency reduction |
|---|---:|---:|
| All options disabled | 10.47168 | — |
| Early slot release | 10.39029 | 0.777% |
| Release-only counters | 10.37975 | 0.878% |

The initial atomic-load/store queue measured 10.43274 ms/token; its combination with release-only counters measured 10.39643. **Both are excluded as validated candidates:** racecheck rejected the shared-memory handoff. Their data and failure logs are preserved. The handoff was replaced with hardware mbarriers and the resulting implementation was benchmarked again below.

The release-only block comparisons ranged from a 1.14% regression to a 1.70% improvement. Early release improved block medians by 0.211%, 0.024%, and 0.795%. These small, variable results motivated confirmation against the untouched production binary, rather than treating the first pooled median as a robust gain.

## Confirmation against untouched production

A separate, balanced three-block comparison used the original production binary, release-only counters, and actual batched publication. Each configuration has another 30 full requests.

| Implementation | Median ms/token | p95 ms/token | Median latency reduction |
|---|---:|---:|---:|
| Untouched production | 10.39468 | 10.47753 | — |
| Release-only counters | 10.35686 | 10.37578 | 0.364% |
| Batched publication | 10.35757 | 10.56374 | 0.357% |

Release-only block median reductions are 0.991%, 0.368%, and 0.336%. Batching regresses 0.662% in its first block, then improves 0.368% and 0.388%; its tail latency is worse. Eliminating nearly 80% of the targeted counter updates did not produce a substantial end-to-end gain.

## Corrected queue against untouched production

After replacing the handoff, another three-block comparison ran 30 requests per configuration. This phase has its own contemporaneous production control.

| Implementation | Median ms/token | p95 ms/token | Median latency change |
|---|---:|---:|---:|
| Untouched production | 10.38867 | 10.40620 | — |
| Hardware-semaphore publication queue | 10.44064 | 11.04112 | **0.500% slower** |

Queue block medians regress 2.053%, 0.457%, and 0.421%. Moving publication to the launcher warp does not help this workload and adds synchronization/queue overhead. Do not merge this design as a latency optimization on this evidence.

## Interpretation

The earlier local storer publication spans do not translate into similarly large recoverable wall time. The real implementations reduce/overlap that work without producing a large end-to-end improvement. This is consistent with substantial overlap and other limits on the critical path; these experiments do not independently measure model DRAM bandwidth or prove one exclusive bottleneck.

Early release and release-only counters are small candidates. Release-only has the cleanest confirmation and only about 38 microseconds median improvement; the extra queue and batching complexity is not justified by the measured latency. No experimental variant is enabled in production by this report.

## Method and evidence

Every timed request performs **100 autoregressive decode steps**, batch one, BF16, context 1024 after prefill, using the same local Llama 1B weights and hashed prompt. Five full warmup requests precede ten measured requests in each process. Timings include embedding, host/runtime dispatch, native decoder, head, argmax, and final token IDs copied to CPU; they exclude loading, prefill, cache restoration, and reference checks. No observer timers or profiling tools run inside the performance measurements.

There are **300 timed requests / 30,000 timed decode steps** across all phases, including the two initial queue versions subsequently excluded. Of these, 240 requests / 24,000 steps belong to the remaining candidates and their controls. All 30 process-level first-step reference checks pass. Accepted numerical tolerances were not retuned; full generated-token trajectories remain informational, as in the existing benchmark.

All builds retain 96 registers, 400-byte stack/local allocation, 4-byte spill stores/loads, 10,384 static shared bytes, 214,015 dynamic shared bytes, 640 threads per CTA, and one resident CTA per SM. Compiler flags, source fingerprints, binary hashes, commit IDs, patches, and build logs are retained in [provenance.json](provenance.json). Source/binary fingerprint checks run in every benchmark process. System clocks, power mode, and unrelated system load were not controlled; unrelated NVFP4 compilation in the shared workspace was left untouched. Interpret sub-percent results within those limits.

Raw results: [first comparison](summary.json), [confirmation](confirmation/summary.json), [corrected queue](queue-v2/summary.json), and [static update counts](batched-static-counts.json). Each directory includes request samples and execution order. The final hardware-semaphore queue and batched implementation each pass full-decoder memcheck and synccheck with zero errors and racecheck with zero hazards. Results and the preserved initial queue failure are in [sanitizers](sanitizers/).

Implementation branches:

- `experiment/thor-output-protocol`: early release, release-only counters, initial queue (superseded).
- `experiment/thor-batched-publication`: consecutive-counter aggregation.
- `experiment/thor-publication-queue-v2`: final hardware-semaphore queue.

The exact source patches are [implementation.patch](implementation.patch), [batching.patch](batching.patch), and [queue-v2.patch](queue-v2.patch). Apply the initial implementation patch first, then the relevant follow-up patch. Scripts under `tools/analysis/output_protocol` build the separate binaries, execute the real autoregressive benchmark, and regenerate aggregates. Run `python3 tools/analysis/output_protocol/report.py` to regenerate all three summaries from the saved requests.
