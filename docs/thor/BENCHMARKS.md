# Benchmark evidence

Measured context-1024 BF16 same-cache one-token replays are in [benchmark-context1024.md](benchmark-context1024.md), with raw JSON/CSV for native, PyTorch eager/graphs and Python VM. These are exploratory measurements under uncontrolled existing load. They do not establish an autoregressive speedup, bandwidth, energy, or model-wide numerical parity.

Required protocol: separate prefill, decode and generation; at least 5 warmups and 30 synchronized event-timed repetitions, 3 independent batches for finalists. Match model, BF16 precision, context, batch, power profile and contention. Save raw samples and all failures; unavailable metrics must be null. Internal clock64 measurements describe worker-relative intervals only. Capture current clocks/power and tegrastats without changing power settings. Contexts 128/1024/8192 and 1/100-token decode require model-supported memory checks first.

Host validation covers scheduling, serialization, page layouts, benchmark reporting and numerical gate behavior. Native synthetic and model-backed checks are recorded separately; host tests alone do not establish CUDA correctness.

## Harness interface

```bash
python tools/thor/bench.py --dry-run --output local/preflight.json
python tools/thor/report.py local/preflight.json
python tools/thor/sweep.py --output local/thor-sweep
```

The sweep defaults to planning. `--execute --adapter module:factory` requires an explicit adapter; the runner does not fetch models. The adapter exposes `metadata`, `reset()`, `reference()`, `run()`, and `check(output, reference)`. `check` must return `passed` and numerical error statistics. `reset` restores matched input/KV state outside timing; reference outputs must be independent snapshots. All auxiliary CUDA streams must join the current stream before `run` returns. Metadata identifies model, precision, context, batch, scope (prefill/decode/generate/kernel/robotics_block), variant, power profile and contention. Scope-specific model adapters, complete metric schema, power-log integration and traces remain to implement.

Sweep adapters receive `THOR_CONFIG` JSON and `THOR_BINARY_DIR`; they must apply the exact compiled config, scheduler and worker count and include it as `metadata.config`. The planner rejects unavailable layouts and architectural/precision variants, and compile-time register overcommit. Builds are cached by full configuration, toolchain versions and source contents. Builds/trials run in process groups with hard timeouts. Compile failures are retried; pilot candidates use 10 samples and 1 batch. Top three pilots receive the standard 30 samples/3 batches. Current selection uses p50 and still needs matched-profile comparison enforcement, Pareto objectives and statistical significance before its selection can be a recommendation.

Host checks now also exercise timing-stat validity, failure reporting, sweep constraint filtering, cache invalidation and process timeout handling. These do not validate CUDA-event or kernel behavior. `benchmark-dry-run.json` is preflight evidence only.

## Current native evidence

Model-free smoke and native compile reports are linked in THOR_RESULTS.md. They are correctness/resource evidence only. Nsight Systems traces can be captured; Nsight Compute counters are denied by current permissions. The initial register-support diagnosis was invalid: generic-PTX validation from `-arch` caused the failure. Explicit architecture-specific `-gencode` builds support dynamic register redistribution, including on-device Thor checks. Both static caps and dynamic role budgets can be investigated, subject to actual occupancy/resource/numerical checks.

## Synthetic correctness without weights

```bash
source tools/thor/env.sh
python tools/thor/synthetic.py --op qkv --output local/qkv-check.json
# Also available: noop, o_proj, down, upgate, lm_head
```

The parent process enforces a timeout. Fixtures generate seeded BF16 inputs/weights locally and precomplete global dependencies; they verify selected output slices using a PyTorch FP32 accumulation reference and explicit tolerances. No model weights are fetched. This path currently requires the default verified 20-worker/16-warp/96-register-cap build and checks compiled resource/config metadata. It deliberately does not attest arbitrary configurations for tuning. These fixtures are not trained-model decode benchmarks.

## Numerical gates and repeatability

Following user direction, full-model logit NRMSE now defaults to 3%; hidden/KV NRMSE stays at 2%, and pointwise maximum-error checks are unchanged. `--logit-nrmse-limit` records an explicit alternative. Historical 2% failures remain unchanged. Every repeated correctness trial must pass; next-token agreement is recorded separately. The ordered reduction experiment produced repeatable but worse raw errors in one context, so it is not evidence for relaxing gates further. Mean logit bias, centered error and KL diagnostics are recorded separately and do not replace raw logit acceptance.

## vLLM and autoregressive comparison

vLLM 0.32.0 uses a separate ARM64 environment and the same local Llama-3.2-1B-Instruct weights, BF16, batch/concurrency one, shared 1024-token prompt and greedy fixed-length output. Prefix caching and speculative decode are disabled. Engine TPOT uses first/last-token engine timestamps, with TTFT and wall end-to-end latency recorded separately. The native autoregressive runner excludes loading/prefill and restores the cache before each request, then generates 100 successive tokens. Host dispatch and final token transfer are included. These scopes differ from the CUDA-event one-token replay table.

```bash
python3 tools/thor/setup_vllm.py
source tools/thor/vllm_env.sh
python -m tools.thor.vllm_bench --disable-flashinfer-autotune --output docs/thor/bench-vllm-autoregressive.json
# Native: use a fresh build after any source/tool changes.
source tools/thor/env.sh
make -C demos/low-latency-llama GPU=THOR
timeout 600 python -m tools.thor.autoregressive_bench --engine native --output docs/thor/bench-native-autoregressive.json
```

Initial vLLM startup failures are preserved as `vllm-smoke*.json`. A failed startup has no latency result. Disabling FlashInfer autotuning is an explicit selectable workaround and does not disable CUDA Graph execution. cuRAND headers are exposed through a selective overlay to avoid mixing pip CUDA 13.0 runtime headers with the CUDA 13.2 JIT compiler.

The final optimized vLLM run completed 5 warmups and 3 batches of 30 requests: median request-mean TPOT 11.640 ms, p95 11.685 ms, p99 11.697 ms; median TTFT 46.539 ms. See `bench-vllm-autoregressive.json`. Full and piecewise CUDA Graph capture remained enabled; FlashAttention 2 was selected, FlashInfer autotuning disabled. An earlier complete run overlapped an agent diagnostic and is retained as `bench-vllm-autoregressive-initial.json`, excluded from the primary comparison.

The complete sequential native/vLLM comparison is [benchmark-autoregressive.md](benchmark-autoregressive.md), with [raw request CSV](benchmark-autoregressive.csv). Timing protocol completion is distinct from native sequence numerical qualification, which remains open.

Later correctness experiments have separate replay datasets: `bench-reference-rounding-context1024.json` (ordinary launch) and `bench-reference-rounding-cooperative-context1024.json` (cooperative launch with entry grid synchronization). Both use the named combined-rounding variant, and both pass 90 per-repeat numerical checks for this seed/context. Their p50s are 11.832 and 11.866 ms. The launch mode is in `metadata.native_resources.cooperative_launch`; missing means the older ordinary-launch implementation. Background load is uncontrolled, so the small difference is not a measured causal overhead. Neither result replaces the original 100-token native/vLLM dataset or closes the broader numerical acceptance gate.

## Pipeline-stage experiments

```bash
source tools/thor/env.sh
python -m tools.thor.stage_matrix --config configs/thor/reference-rounding.json --benchmark --output local/stage-comparison-run1
# Fix all page budgets at 13 to isolate staging from allocation footprint:
python -m tools.thor.stage_matrix --config configs/thor/reference-rounding.json --pages 13 --benchmark --output local/staging-only-run1
# Model-free sanitizer matrix; timing is optional and never run under the sanitizer:
python -m tools.thor.stage_matrix --config configs/thor/reference-rounding.json --sanitizer racecheck --output local/stage-racecheck-run1
```

The initial `stage-matrix/` run used the historical BF16 atomic down-projection. All opcode checks passed in every layout, but each full-model repeated batch encountered a numerical failure; those failures are retained with no accepted latency summaries. One-stage V-cache NRMSE reached 2.038% against its unchanged 2% gate. `stage-matrix-reference-rounding/` was explicitly interrupted after identifying an isolated-fixture dependency-reset bug, not classified as a production pipeline deadlock. The corrected fixture has host regression tests; subsequent complete evidence is recorded in `stage-matrix-qualified/`.

The completed [stage ablation table](benchmark-stage-ablation.md) includes five 90-sample replay datasets: minimum footprints for 1/2/3 stages, plus 1/2 stages held at the compatibility 13-page budget. The raw reports retain numerical checks and compiler/resource metadata; companion CSV and SVG contain the same verified measurements. Variant order was sequential and external load/clocks uncontrolled. Keep the three-stage compatibility default pending broader accuracy and controlled tuning; treat the one-stage footprint as an experimentally validated candidate, not a demonstrated end-to-end speedup.
