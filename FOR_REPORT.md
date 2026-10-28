# Rethinking Megakernels for Jetson Thor

**Report snapshot: 28 October 2026.** This report reviews the native Thor port, framework/runtime experiments, and native NVFP4 implementation. Measurements come from the linked repository artifacts; no new GPU benchmark was run to prepare this document. The evidence snapshot includes native W4A4 commit `5f5aec3`, W4A16 commit `e5c8264`, the selective L2-cache experiment `d049d54`, and W4A8 commit `db71c7e` on `feat/thor-mixed-fp4-fp8`. The validated W4A8 backend, documentation and raw results are now integrated into `feat/thor-megakernels`; the originating branch is preserved for provenance. Recorded results are distinguished from subsequent working-tree edits.

## Findings

- A native Jetson Thor port of the original persistent BF16 megakernel works and measures approximately **10.40 ms/token** for batch-one Llama-3.2-1B-Instruct decode. Its initial comparison measured **8.37% lower median latency than BF16 vLLM** under the recorded protocol.
- The investigated BF16 framework/runtime optimizations yield **sub-percent gains**, not a large optimized-port speedup. Release-only readiness counters improve a confirmation median by **0.36%**. Moving readiness publication to another warp is **0.50% slower**.
- A native W4A4 NVFP4 backend integrated into the actual VM measures **5.13 ms/token**, approximately **2.03× faster than the BF16 port**. This combines lower precision, a different compute backend, and runtime changes; it is **not a framework-only speedup or a quality-equivalent comparison**.
- The mixed **W4A8 MXFP4/MXFP8** backend measures **5.040 ms/token**, versus **5.684 ms W4A4** and **10.938 ms native BF16** in its fresh five-request controls. Its recorded subset perplexity increases **15.88%**. Both weights' format/scales and activation precision change, so the quality difference cannot be attributed to activations alone.
- The NVFP4 model has a measurable quality cost: conditional perplexity on the recorded 1,024-token subset increases **20.16%**. The weight-only W4A16 path reduces that degradation to **10.66%**, but its recorded implementation is much slower than BF16.

The central lesson is that edge adaptation must change the balance between memory traffic, compute granularity, and resource ownership. Persistence alone cannot eliminate the cost of repeatedly reading weights, and large local waits do not necessarily indicate large recoverable end-to-end latency.

## 1. Rethinking the implementation and its limitations on edge devices

### 1.1 Thor is not a proportionally smaller B200

| Property | Thor used here | B200 context | Design implication |
|---|---|---|---|
| Architecture / target | Blackwell, CC 11.0, native `sm_110a` | Blackwell; original repository B200 target | Shared architecture branding does not make target code, launch policy, or resource assumptions portable. |
| SM count | 20, measured | 148 in the original demo's configuration | Task granularity, worker indexing, instruction sizes, and cooperative admission must adapt. The B200 count here is a repository target, not a measurement on a B200. |
| External-memory bandwidth | Advertised 273 GB/s | Up to 8 TB/s | The bandwidth reduction is approximately 29×, much greater than the 7.4× reduction in configured SMs. |
| Memory system | 128 GB shared LPDDR5X | 180 GB HBM3e in the cited HGX configuration | Shared capacity does not imply on-chip residency; CPU/GPU system traffic also matters on an integrated device. |
| Registers per SM | 65,536 32-bit registers, measured | 65,536 | Fewer total SMs do not mean a proportionally smaller per-SM register budget. |
| Shared memory per SM | 228 KiB, measured | 228 KiB | Occupancy cannot be inferred from a generic “edge has less shared memory” assumption. |
| Resident warps per SM | 48, measured | 64 | Warp specialization and service-warps consume a different fraction of the available scheduling capacity. |

Thor specifications are from [NVIDIA's Thor product page](https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-thor/). B200 memory/bandwidth are from [NVIDIA's HGX reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html); per-SM B200 limits are documented in the [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html). Thor observations are recorded in the [local inventory](docs/thor/enabling-port/inventory.json) and diagnostic records.

Using those bandwidth figures and SM counts, Thor has roughly **four times less external-memory bandwidth per SM**. This is a design inference, not an application performance measurement. It motivates reducing bytes per operation and matching staging to the memory system, rather than transferring B200 scheduling choices unchanged.

Thor's supported low-precision Tensor Core instructions remain useful. The initial port's use of a Hopper/TMA compatibility path was an implementation choice, not evidence that Thor lacks TMEM or `tcgen05` support.

### 1.2 What the native port actually changed

The enabling port preserves the original BF16 SIMT matvec arithmetic and static round-robin scheduling. Its changes make native execution possible and correct:

| Change | Purpose | Classification |
|---|---|---|
| Native SM110a build and project-local ARM64 dependencies | Produce executable native Thor code and resolve host/toolchain compatibility | Thor enabling |
| Logical `blockIdx.x` worker IDs | Avoid using sparse physical SM IDs as queue indices | Generic runtime portability |
| Cooperative launch, entry synchronization, compiled occupancy/resource checks | Ensure all persistent dependency participants can be resident; reject unsafe launch configurations | Runtime correctness/admission |
| Static 96-register cap for the original 640-thread CTA | Fit its register demand within the per-SM budget | Thor resource configuration |
| Per-thread semaphore participation and compiler memory clobbers | Make pipeline/instruction publication and observation reliable | Synchronization correctness |
| Explicit global acquire/release publication; TMA completion and page-reuse fixes | Prevent consumers observing incomplete results or reused storage | Runtime correctness |
| Bounded instruction serialization and payload splitting | Handle larger per-worker payloads when work is distributed over fewer SMs | Generic scheduling/serialization principle, integrated into the demo |

Attention ring-phase, empty-partition, and initialization fixes also exist. They are operator correctness repairs, **not Thor-specific framework optimizations**. Llama-specific arithmetic changes and earlier speculative tuning were removed from the enabling baseline and retained separately in the historical archive.

The BF16 configuration remains 16 consumer warps plus four service warps, 640 threads/CTA, three weight stages, thirteen 16-KiB pages, and one resident persistent CTA per SM. There is no meaningful speedup ratio against an unmodified B200-targeted executable on Thor: that is not a valid running baseline. Performance optimization must be compared against the **working enabling-only port**.

Evidence: [bring-up scope](docs/thor/BRINGUP.md), [configuration](include/config.cuh), [checked binding](include/checked_binding.cuh), and [global publication](include/gmem_barrier.cuh).

### 1.3 Persistence hides latency; it does not eliminate weight traffic

The checkpoint contains approximately **2.47 GB of BF16 parameters**, far exceeding the measured **32 MiB L2**. Loading weights into device memory once avoids loading them from storage each token. It does not keep the entire model in registers, shared memory, or L2: subsequent decode steps still read weights from LPDDR into on-chip storage.

A persistent loader can prefetch a tile while consumers compute another tile. Other SMs can issue work concurrently. That overlap reduces exposed latency and helps use bandwidth, but it does not remove the aggregate bytes crossing the memory interface.

The steady-state trace measured:

| Quantity | ms/token |
|---|---:|
| Unprofiled decode probe | 10.4002 |
| Persistent megakernel in the trace | 10.3329 |
| Other traced GPU work | 0.0432 |
| Time without traced GPU activity | 0.0228 |

The megakernel occupies approximately **99.37%** of the marked interval. Python or launch elimination therefore has little exposed idle time to recover in this workload. CPU API durations overlap GPU execution and must not be added to it.

Source-level accounting covers 3,156 tasks and **2.523 GB requested payload per decode**. Distinct linear-weight reads account for **97.97%**; repeated activation and normalization inputs together account for **0.575%**. These are payload counts, not measured DRAM transactions: caching, polling, atomics, spills, and cache-line effects remain separate.

A matching arithmetic-free TMA probe reads the linear-weight payload in approximately **9.51 ms**, around **260 GB/s**, with one CTA per SM and one issuing lane. More issuing lanes do not materially improve that probe. A shallower stage count helps the probe slightly, but the probe has no arithmetic to overlap; that is not an optimized full-model result.

This supports weight movement as a leading explanation for the BF16 latency. It does **not** prove that every remaining millisecond is removable runtime overhead. In particular, the rough 2.47 GB / 273 GB/s estimate is not a measured “9 ms data movement plus 1.3 ms overhead” decomposition. Hardware traffic/stall counters were permission-blocked, so exact DRAM utilization inside the model remains unverified.

Evidence: [bottleneck investigation](docs/thor/BOTTLENECK.md), [traffic/TMA/wait diagnoses](docs/thor/DIAGNOSES.md).

### 1.4 Generic runtime optimizations: what survived end-to-end testing

The output protocol looked promising because sampled consumers spent substantial time waiting for output slots. Follow-up instrumentation attributed much of the QKV/up-gate store-call span to global readiness publication. The percentages were local, overlapping worker timings—not fractions of token latency.

Actual implementations tested that lead:

- **Early slot release:** return shared scratch after TMA completes, before global readiness publication.
- **Cheaper publication:** use release-only counter RMWs while retaining acquire readers.
- **Fewer publications:** combine consecutive updates to the same readiness counter and flush at counter/instruction boundaries.
- **Decoupled publication:** hand completion records to the existing launcher warp through a per-instruction queue.

These change completion, synchronization, and resource lifetimes, not model arithmetic. The experimental call sites target QKV/up-gate because those paths exposed the backpressure; the underlying protocol is reusable. The evidence is measured on Thor, but no matching B200 experiment establishes that the costs are unique to Thor.

The first queue handoff failed racecheck and was replaced with hardware ready/consumed mbarriers. Its initial timings are preserved but excluded as validated candidates. The corrected queue and batching implementations pass full-decoder memory, synchronization, and race checks.

**Lesson:** local backpressure is useful for finding experiments, but only full decode latency determines whether an optimization wins. Reducing targeted publication calls by nearly 80% did not produce a substantial end-to-end improvement; queueing also introduced its own overhead and delayed work.

## 2. Benchmark: simple Thor port versus runtime-optimized Thor port

### 2.1 Same-precision framework/runtime comparison

All rows below use BF16 Llama-3.2-1B-Instruct, batch one, context 1024, and 100 autoregressive decode steps per request. Each configuration has three blocks of ten measured requests, with five full warmup requests before each block/process. Each row uses its own comparison cohort; do not combine baselines from different phases.

| Runtime experiment | Its baseline ms/token | Candidate ms/token | Observed median change | Interpretation |
|---|---:|---:|---:|---|
| Early output-slot release | 10.47168 | 10.39029 | 0.777% lower | Small initial result; not separately confirmed against the untouched executable |
| Release-only counters, confirmation | 10.39468 | 10.35686 | **0.364% lower** | Approximately **37.8 µs/token** saved; simplest confirmed candidate |
| Batched counters, confirmation | 10.39468 | 10.35757 | 0.357% lower | One block regresses and p95 worsens; no additional clear benefit |
| Corrected publisher queue | 10.38867 | 10.44064 | **0.500% higher** | Slower in all three blocks; reject as a latency optimization for this workload |

Release-only confirmation block median reductions are 0.991%, 0.368%, and 0.336%. System load and clocks were uncontrolled, so sub-percent effects should not be presented as a large or universally robust gain.

There is currently **no merged BF16 “optimized Thor port” demonstrating a substantial gain over the simple port**. The table describes real experimental implementations, with the enabling-only path kept available. The meaningful same-precision result is a small publication-policy improvement, not a redesigned runtime that materially outperforms the baseline.

The complete investigation ran 300 timed requests / 30,000 decode steps, including the initial queue versions later excluded. All samples, code patches, fingerprints, branches, and validation results are in [the implementation report](docs/thor/output-protocol-results/RESULTS.md).

### 2.2 Enabling-only native port versus vLLM

The original matched BF16 comparison used 90 requests per engine:

| Engine | Median ms/token | p95 request-mean ms/token |
|---|---:|---:|
| Native Thor enabling port | 10.395 | 10.413 |
| vLLM 0.32.0, BF16, graphs enabled | 11.345 | 11.393 |

The native port has **8.37% lower observed median decode latency**. Both use the same local weights and hashed prompt. Native timing includes dispatch, embedding, decoder, head, greedy selection, and final output-ID copy, excluding loading/prefill/cache restoration. vLLM timing uses engine first/last-token timestamps and includes its scheduling/sampling. Runtime-boundary differences and uncontrolled load limit interpretation; this is a workload comparison, not a pure kernel speed ratio.

Evidence: [benchmark protocol and artifacts](docs/thor/BENCHMARKS.md).

## 3. First native NVFP4 megakernel in this project

“First” here means the first native NVFP4 integration in this repository's implementation sequence. A worldwide priority claim has not been established. Native FP4 Tensor Core instructions and CUTLASS support predate this work; the contribution is their integration into this persistent VM on Thor.

### Model-agnostic optimization, model-specific integration

The optimization is a **model-agnostic, format-aware low-precision execution backend**. It changes operand representation, data movement, compute instructions and resource lifetimes without changing model architecture. Llama is the integration and evaluation workload, not the definition of the backend.

- **Operand descriptors** carry dimensions, packed weights, scale layouts, input/output pointers and precision modes. Static weights are packed once at setup and remain compressed in device memory.
- **Reusable linear operators** stage operands and scales with TMA, execute native Tensor Core MMA, and return BF16 outputs from FP32 accumulations.
- **Runtime resource management** owns TMEM across VM instructions and synchronizes pipeline stages, shared-memory reuse and result publication.
- **Activation preparation** handles changing inputs on GPU, with producer-epilogue quantization where compatible and explicit preparation at nonlinear boundaries.

The model adapter maps layers, tensors, normalization, gating, attention and dependencies into those instructions. Another supported model can reuse the backend mechanisms but still needs an adapter, shape support and validation. W4A4 and W4A8 require their own formats, packing and scaling; they are not interchangeable dtype flags.

Thus the **implementation is model-agnostic, while the measured latency and quality outcomes remain model-dependent**. The approximately 2× improvement combines precision, backend and runtime changes; existing measurements do not isolate those contributions or establish the same gain for other models.

### 3.1 Change the representation before streaming

NVFP4 stores E2M1 four-bit values, an E4M3 scale per 16 values, and a tensor-level scale. Including block scales, canonical storage is **0.5625 bytes/value**, approximately **3.56× smaller than BF16** before layout padding. See [NVIDIA's format description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).

Static weights are quantized and packed during setup, then retained compressed in device memory. The recorded model's quantized linear weights occupy **695,107,584 bytes**, versus **2,471,493,632 BF16 bytes**. Quantizing only after streaming BF16 weights into shared memory would not provide this off-chip traffic reduction.

Activations change each token. W4A4 therefore prepares and quantizes activations on GPU, including block scales. The recorded implementation uses a fixed activation tensor scale of 1/64 with dynamic block scales; it does not establish a general calibrated or dynamically chosen global-scale policy.

The format must be represented explicitly in the framework: packed values, physical scale layout, alignment, activation buffers, output precision, and stage lifetimes are distinct resources. A canonical 16×2048 weight stage falls from **64 KiB BF16 to 18 KiB NVFP4 including block scales**, but this does not mean every shared-memory page should shrink by the same ratio. Native scale layouts can require padding/interleaving, and attention/scratch/output storage retains its own needs.

### 3.2 Native execution required runtime integration, not a wrapped GEMM

The first cooperative FP4 prototype placed complete standalone GEMM bodies inside a persistent instruction loop. It was correct as a mechanism experiment, but slower than an equivalent FP4 CUDA graph:

| Eight dependent synthetic linear operations | Graph µs | Persistent prototype µs |
|---|---:|---:|
| Width 2048, automatic seven-stage case | 80.22 | 216.99 |
| Width 4096, automatic seven-stage case | 448.00 | 671.68 |

These are synthetic chains, not full-model timings. Two-stage storage also failed to improve that prototype. The result demonstrates that persistence can lose to a graph when resource management and operator scheduling are integrated poorly; it does not establish that graphs always beat a full megakernel.

The subsequent native backend uses the real VM's loader, launcher, consumer, storer, instruction loop, and page lifetimes. CUTLASS supplies layouts, descriptors, and instruction atoms; its standalone kernel scheduler is not executed inside the VM.

Important framework changes include:

1. **VM-lifetime TMEM ownership.** The launcher allocates TMEM once for a VM launch and relinquishes/deallocates it after all workers finish. Reusing a standalone-kernel allocation lifecycle inside a persistent loop had been invalid after relinquishing the allocation permit.
2. **Separate asynchronous lifetimes.** TMA arrival, MMA stage retirement, accumulated-tile readiness, accumulator reuse, and output publication use explicit synchronization. A stage is not reusable merely because its load finished.
3. **Format-aware descriptors and staging.** Packed operands and scales travel together through TMA and native block-scaled `tcgen05` MMA. Accumulation is FP32; outputs remain BF16.
4. **Backend-specific resource configuration.** The native FP4 configuration uses four consumer warps plus four service warps, 256 threads/CTA, two 64-KiB pages, and a 512-column TMEM allocation. It checks exactly one resident CTA/SM. This replaces the BF16 backend's configuration for this path, rather than proving one universal page/warp setting is optimal.
5. **Explicit activation preparation.** Generic producer-epilogue quantization is available for compatible dependent linear operations. The full model still uses explicit preparation around normalization and nonlinear gating; it does not claim every conversion is fused away.

A 128-row MMA tile also matters on a 20-SM device: a 2048-output operation has only 16 such tiles. Coarser Tensor Core work can leave workers without an output tile, and batch-one matrix-vector work cannot be assumed to achieve advertised matrix throughput. Smaller shared-memory allocation alone cannot overcome TMEM ownership, registers, or insufficient work granularity.

Implementation: [TMEM lifetime](include/megakernel.cuh), [native linear operator](include/ops/nvfp4_linear.cuh), [format/type layouts](include/ops/nvfp4_types.cuh), [experimental model adapter](experimental/native_fp4_vm/model.cu), and [format-storage planner](experimental/quantized_runtime/format_storage.cuh). The model adapter remains specific to the local 16-layer Llama configuration; the reusable contribution is the backend/runtime mechanism.

### 3.3 Full-model latency and the precision trade-off

The later recorded model comparison uses batch one, context 1024, 100 autoregressive steps, five warmups and 30 requests per engine:

| Path | Linear operands | Median ms/token | Relative to native BF16 |
|---|---|---:|---:|
| Simple native Thor port | BF16 | 10.3952 | 1.00× |
| Native NVFP4 VM | W4A4 | **5.1297** | **2.03× faster** |
| vLLM with graphs | BF16 | 11.3656 | Separate serving baseline |

W4A4 has **50.65% lower observed latency than native BF16**, and **54.87% lower than the recorded BF16 vLLM run**. These ratios are calculated from this later cohort, not the earlier vLLM table. Weight packing, loading and prefill are excluded; dynamic activation preparation and actual decode work are included. Timing ends with generated IDs on CPU.

This is an implementation-system result combining compressed weight traffic, native low-precision compute, a different VM configuration, and quantization. There is no factorial ablation proving how much each contributes. It must not be reported as a 2× gain from model-independent BF16 scheduling changes, nor as a comparison with NVFP4 vLLM. No energy-per-token result was collected.

Evidence: [native FP4 report](docs/thor/NATIVE_FP4_VM.md), [BF16 raw data](docs/thor/native-fp4-vm/bf16-e2e.json), [W4A4 raw data](docs/thor/native-fp4-vm/fp4-e2e.json), [later vLLM raw data](docs/thor/native-fp4-vm/vllm-bf16-e2e.json).

### 3.4 Correct execution and retained model quality are different questions

Frozen-input checks isolate implementation errors from quantization changes: all 65 activation conversions match their independent reference bit-for-bit, and linear outputs agree with decoded FP32 products to within 0.0055% relative RMS error. With independently decoded linear products and the same unchanged VM attention backend, full-model logits NRMSE is approximately **0.00091%**.

An oracle using a different attention implementation gives larger accumulated differences because small attention differences can cross FP4 rounding boundaries. Matching the attention backend makes the quantized-execution reference meaningful; it does not establish equivalence to the original BF16 model. Original-BF16 tolerance checks fail for the quantized model and are recorded, rather than hidden.

On the deterministic WikiText-2 subset, both paths use 128-token BF16 prefixes followed by 64 teacher-forced decode targets per window, for 16 windows / 1,024 scored tokens:

| Path | Conditional perplexity | Increase versus BF16 | Top-token agreement with BF16 |
|---|---:|---:|---:|
| BF16 | 18.4948 | — | — |
| Native W4A4 | 22.2238 | **20.16%** | 78.42% |
| Mixed W4A8 | 21.4314 | **15.88%** | 80.27% |
| Experimental W4A16 | 20.4661 | **10.66%** | 85.45% |

This is not standard full-corpus perplexity. Eight authored prompts also expose changed answers and a grammar regression; they are a sanity check, not a broad task evaluation. The speed result therefore has a real, measured quality trade-off.

Full-model memory/synchronization/race checks were recorded for the W4A4 build before later diagnostic tracing additions; subsequent source changes require their own qualification. The recorded W4A16 build also passes these checks. Historical artifacts qualify their recorded binaries, not arbitrary current edits.

Evidence: [matched quantized reference](docs/thor/native-fp4-vm/model-shared-attention-reference.json), [W4A4 subset quality](docs/thor/native-fp4-vm/wikitext-subset-quality.json), [W4A16 subset quality](docs/thor/w4a16-vm/model-corpus.json), [W4A8 subset quality](docs/thor/w4a8-vm/model-corpus.json).

### 3.5 Weight-only FP4 is not automatically faster

The W4A16 path keeps activations in BF16, loads compressed weights and scales through TMA, decodes weights into on-chip BF16 buffers, and uses BF16 `tcgen05` MMA. It does not materialize a full BF16 weight copy in global memory.

Its recorded double-buffer implementation nevertheless measures **53.22 ms/token** in a five-request, 100-token run, compared with the earlier approximately 10.40 ms BF16 and 5.13 ms W4A4 runs. These are different cohorts/protocols; even so, the slowdown is substantial. Double buffering did not demonstrate a win. Reduced off-chip bytes were outweighed by the conversion/staging implementation.

Frozen-input linear checks agree closely with decoded-weight FP32 products, but the full-model shared-attention oracle still has **2.68% logits NRMSE** that is not fully explained by those checks. This path should not be described as a quality-equivalent or production-ready replacement.

This is an implementation-specific negative result, not proof that W4A16 is inherently slow. The measured mixed-format follow-up is described below; subsequent model-adapter edits are not retrospectively qualified by these artifacts. See [the recorded W4A16 report](docs/thor/W4A16_VM.md).

### 3.6 Selective L2 retention is useful only within its budget

A subsequent standalone experiment predecodes selected weights once and applies a persisting-L2 access policy to their BF16 buffers. Thor reports **32 MiB L2**, a **24 MiB maximum persisting set-aside**, and a **128 MiB maximum access-policy window**. A policy window is not a guarantee that all its bytes remain cached.

For an 8-MiB decoded matrix, the recorded primitive falls from **323.42 µs with an ordinary BF16 cache to 147.30 µs with persisting policy**, after an intervening 128-MiB memory pass. The 128-MiB matrix shows no benefit: **5483.36 versus 5489.90 µs**. Outputs are bit-identical. These controls use the original unpipelined W4A16 primitive, not the newer VM operator; the figures are **not model or VM end-to-end improvements**.

The graph kernel node needs its own access-policy attribute before instantiation: applying the policy only to the ordinary stream was unreliable with PyTorch's separate capture stream. This is a reusable framework/graph integration lesson. The complete decoded model is approximately 2.5 GB; a 24-MiB retained subset covers only about 1% of its decoded weight bytes. Predecoding everything into LPDDR would also trade away compressed weight streaming.

Evidence: [L2-cache experiment and raw measurements](docs/thor/W4A16_L2_CACHE.md). No full-model speedup from selective retention has been established.

### 3.7 Mixed W4A8: native FP4 weights with FP8 activations

The measured `feat/thor-mixed-fp4-fp8` backend at commit `db71c7e` uses native `tcgen05.mma.kind::mxf8f6f4`: E2M1 weights, E4M3 activations, FP32 accumulation and block-32 UE8M0 scales. **This is MXFP4/MXFP8, not NVFP4 block-16 scaling.** Weights are independently quantized once during setup; the W4A4 packed weights are not reused unchanged.

The VM loader stages compressed weights, FP8 activations and scales through TMA. The launcher issues mixed MMA directly, avoiding W4A16's software conversion of weights into BF16 tiles. Two 64-KiB pages hold the two-stage operator pool. Normalization, gating, RoPE, attention, output and residual boundaries retain BF16/FP32 arithmetic. This is a reusable precision backend integrated into the same VM roles, not a Llama-specific arithmetic rewrite or a Thor-exclusive instruction.

#### End-to-end latency

Llama-3.2-1B-Instruct, batch one, context 1024, 100 autoregressive decode steps, two warmup requests and **five measured requests per run**. Prompts have the same recorded input hash. Each row below is from this W4A8 comparison cohort, rather than the earlier 30-request W4A4 cohort.

| Backend | Median ms/token |
|---|---:|
| Mixed W4A8, initial run | 5.0404 |
| Mixed W4A8, repeat after controls | **5.0397** |
| Native W4A4, fresh control | 5.6837 |
| Native BF16, fresh control | 10.9376 |
| vLLM BF16, graphs enabled, fresh control | 11.9377 |

Using the repeated mixed run, W4A8 measures **11.33% lower latency than the fresh W4A4 control**, **2.17× faster than native BF16**, and **2.37× faster than BF16 vLLM**. This is not a precision-matched framework comparison. The five-request count is small, and clocks/system load remain uncontrolled; do not substitute the earlier 5.1297-ms W4A4 run into this comparison.

Packing, loading and prefill are excluded; activation preparation, embedding, VM execution, barriers, argmax and final output-ID transfer are included in native decode timing. vLLM retains its engine timestamp boundary, graphs enabled and FlashInfer autotuning disabled. An autotuning-enabled startup failed with illegal memory access and is archived; it is not used as a performance baseline.

#### Correctness and quality

Against independently decoded FP32 linear products with the same quantized formats and VM attention, BF16 logits, hidden state and KV caches match exactly on both seed 1/context 1024 and seed 2/context 128. Frozen-input checks cover all 65 linear operators: prepared activations match exactly, and maximum linear-output NRMSE is **0.00255%**. Full-model memcheck, synccheck and racecheck pass with zero errors/hazards. W4A4 remains selectable and retains its matched-reference check; W4A16's reference check is unchanged.

Against original BF16 weights/arithmetic, first-decode logits NRMSE is **19.04%**. On the same 1,024 teacher-forced WikiText tokens, conditional perplexity is **21.4314 versus 18.4948 BF16 (+15.88%)**, with **80.27%** top-token agreement. This sits between the recorded W4A4 (+20.16%) and W4A16 (+10.66%) perplexity increases. Both weight quantization/scaling and activation precision differ from W4A4, so these measurements do not isolate activation precision's contribution.

The authored prompt check recovers the correct grammar word but still produces a changed, questionable ice explanation. The subset and eight prompts do not establish quality equivalence or broad task accuracy.

**Lesson:** mixed native FP4/FP8 is a more promising measured latency/quality compromise than this project's software-dequantizing W4A16 implementation. It retains compressed weights without the BF16 expansion loop and improves the recorded subset quality relative to W4A4. It still changes model behavior, and its advantage needs broader quality testing and longer performance runs before being treated as a deployment result.

Evidence: [W4A8 implementation report](docs/thor/W4A8_VM.md), [initial timing](docs/thor/w4a8-vm/model-e2e.json), [repeat timing](docs/thor/w4a8-vm/model-e2e-repeat.json), [fresh W4A4 control](docs/thor/w4a8-vm/w4a4-e2e.json), [fresh BF16 control](docs/thor/w4a8-vm/bf16-e2e.json), [vLLM control](docs/thor/w4a8-vm/vllm-e2e.json), [matched reference](docs/thor/w4a8-vm/model-check.json), [second fixture](docs/thor/w4a8-vm/model-check-seed2.json), [quality](docs/thor/w4a8-vm/model-corpus.json), and [sanitizer process records](docs/thor/w4a8-vm/model-sanitizers.json).

## 4. Lessons to carry into the next edge runtime design

- **Make cooperative residency an admission rule.** Persistent dependencies can deadlock if their producers cannot run. SM count, registers, shared memory and TMEM ownership belong in launch validation.
- **Own asynchronous resources at the right lifetime.** Kernel-scoped allocator assumptions do not automatically survive instruction-level reuse inside a persistent kernel.
- **Design storage around formats and lifetimes.** Packed values, scales, activations, scratch and outputs should not share an implicit “everything is a BF16 page” contract.
- **Use end-to-end latency to select scheduling changes.** Local wait percentages, fewer atomics, smaller allocations and more issuing lanes are explanations or hypotheses—not performance results.
- **Reduce traffic before it crosses LPDDR.** Prequantized resident weights provide a real bandwidth opportunity; quantizing after a BF16 read does not.
- **Keep the precision/quality axis visible.** Native W4A4 and mixed W4A8 provide the large measured latency results, with different formats and quality costs; BF16 runtime tuning is a separate, much smaller result. Kernel correctness, model quality and serving latency require distinct comparisons.
- **Scope the conclusion to measured workloads.** This work covers one trained 1B model at batch one/context 1024 plus synthetic operator chains. It does not establish global Thor optimality, larger-model behavior, throughput scaling, energy efficiency, or a worldwide novelty claim.
