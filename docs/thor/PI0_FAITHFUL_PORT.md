# Exact work required for a faithful BF16 FlashRT port

Source audit: 2026-10-28, FlashRT
`86ef457d16ce5ea490d94493991e76078fcaa97d`.
This is an implementation specification, not a claim that these parts have
been ported. The existing `flashrt_pi0` backend is a partial implementation.

Implementation finding (2026-10-28): the unchanged selected collectives cannot
all inhabit one PTX kernel. Prefix GEMMs/GeGLU use `tcgen05` CTA group 2;
SigLIP FMHA uses group 1. Compiling both into one VM kernel fails with
`uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2) and
that is not allowed`. NVIDIA's PTX ISA requires a uniform CTA group within
a kernel. The faithful implementation therefore needs separate native VM
kernel types composed in a CUDA graph. A strict single-megakernel variant
requires an explicitly different attention or projection collective.
See https://docs.nvidia.com/cuda/parallel-thread-execution/ (tcgen05 granularity).

## Target and definition of fidelity

Target ordinary pi0, batch one, GPU-resident inputs, configurable views/prompt/
action horizon, and ten Euler steps. Use the current matched workload first:
three 224x224 views, 768 visual tokens, zero/16 prompt tokens, 50 actions,
32 action dimensions. Also retain FlashRT Thor's original ten-action case.

The source Pi0 Thor frontend is FP16+FP8, not BF16-only. Its `use_fp8` flag is
stored but is not passed into its forward calls; its decoder has no complete
unquantized branch. Shared Thor vision/prefix code does have an FP16-only path.
Consequently the concrete target must first be a BF16 adaptation of that
unquantized machinery, with an explicitly completed ordinary-pi0 decoder.

Faithful means preserving the selected device collective, layouts, operand
precision, epilogues, rounding boundaries, tile coverage, staging and data
reuse, while replacing host/kernel scheduling with native VM instructions.
It does not mean reproducing cuBLAS's closed-source kernel implementation.
Those sites require an equivalent BF16 device implementation and a measured
comparison with cuBLAS. FP8/FP4, calibration, AWQ, pi0.5 AdaRMSNorm, SM120
skinny kernels, and arbitrary alternative attention algorithms are excluded.

## P0 — A runnable BF16 shadow and a resolved operation manifest

Implement a standalone FlashRT BF16 shadow before changing the native operators:

1. Thread unquantized mode through weight loading and vision, prefix and decoder
   dispatch. Remove Quant, descales, FP8 activation buffers and calibration from
   that path. Replace FP16 storage/conversions with BF16, retaining FP32 GEMM
   accumulation and FP32 normalization/softmax statistics.
2. Parameterize Thor Pi0's hardcoded `Sa=10`, `steps=10`, buffer sizes and cache
   strides. Use the same checkpoint, images, state, token IDs and supplied noise
   as the VM. Declare input BF16-conversion boundaries, language-embedding
   lookup/caching, masks and timestep-generation order identically in both.
3. Complete the unquantized Pi0 decoder using the actual Pi0 sequence:
   state token, action input projection, action/time MLP, ordinary RMSNorm,
   GQA with state masking, final norm, action projection and Euler accumulation.
   Do not substitute pi0.5 conditioning or the existing FP32 folded head path.
4. Record the selected operation sequence and compile-resolved traits: actual
   M/N/K, logical versus padded dimensions, A/B/D types and strides, layouts,
   tile and cluster shapes, stages, threads/warp roles, shared/TMEM storage,
   epilogue and scalar values. Record cuBLASLt algorithm IDs and dispatch choices
   on the real workload. `KernelScheduleAuto` and `StageCountAutoCarveout` are
   compile-time policies, not evidence that three stages were selected.
5. Pin CUTLASS and any FlashRT include overlays as well as the FlashRT revision.
   Produce standalone timing and outputs from this exact BF16 shadow.

Sources: `FlashRT/flash_rt/frontends/torch/pi0_thor.py`,
`frontends/torch/_pi0_thor_spec.py`, `_thor_spec_common.py`,
`models/pi0/pipeline_thor.py`, `hardware/thor/shared_primitives.py`.
The RTX Pi0 unquantized decoder is useful as a control-flow reference, but its
kernel dispatch is not automatically the Thor target.

Deliverables: a BF16 shadow runner/library and a JSON operation/tactic manifest.
The original OpenPI-matching reference remains a separate model oracle.

## P1 — VM support for the actual collectives

The current VM interface is insufficient to copy the clustered collectives
unchanged. Implement these backend-specific capabilities:

| Capability | Required implementation | Current conflict |
|---|---|---|
| Cluster work ownership | One static instruction range owns logical cluster tiles; all participating CTAs receive compatible work and retire it together | Independent CTA streams can advance to different operators |
| 2x2 and 2x1 collectives | A physical cluster configuration plus explicit two-CTA subgroup mapping for two-SM tactics; correct logical ranks and multicast masks | One-CTA kernel and one-CTA TMEM ownership |
| Multidimensional worker IDs | Flatten all grid dimensions for queue, timing and descriptor indexing; distinguish CTA, cluster and pair IDs | `blockIdx.x` alone identifies streams; existing checked binding rejects multidimensional grids |
| Cluster admission | Checked cooperative/cluster launch configuration, cluster occupancy and admitted whole clusters, including descriptor dimensions | The current `workers <= SM count` plus CTA occupancy check is not a cluster-residency proof |
| Role mapping | Map VM loader/MMA/storer/consumers to the collective's load/MMA/epilogue-load/epilogue participants; resolve actual thread counts | All operators inherit the fixed four-consumer/four-service layout |
| Shared storage | An aligned VM-owned arena sized from each collective's actual SharedStorage, including epilogue auxiliary tensors and pipeline barriers | Fixed 13-page allocation and custom TMA tiles are not CUTLASS SharedStorage |
| Register policy | Preserve collective producer/consumer register budgets and legal transitions; account for all persistent threads | Static 255-register cap across every role/operator |
| Named barriers | An explicit mapping/lifetime allocation for VM barriers, CUTLASS reserved barriers and subgroup barriers | Independent hardcoded barrier IDs can collide, as the small-query path already demonstrated |
| TMEM lifetime | One allocator owner, correct group-1/group-2 allocation and addressing, permitted reuse, drained handoffs and final release | Automatic TK allocation must not coexist with a second independent CUTLASS allocator |
| Completion | Drain load/MMA/epilogue pipelines and TMA stores, then publish global row/head counters with release semantics | Existing custom completion semaphores do not describe arbitrary CUTLASS collective completion |

Cluster shape is a kernel launch property; it cannot simply switch from 2x2 to
2x1 on an opcode. A workable one-launch design is a four-CTA physical cluster
with supported logical pair execution for two-SM operators and independent
execution where appropriate. That subgroup implementation must preserve the
collective's rank/address/multicast semantics. If it cannot, cluster geometry
is a real unresolved architecture constraint, not a flag to change silently.
Use the actual device admission result rather than assuming all 20 SMs admit
five simultaneous four-CTA clusters for every resource configuration.

Preserve static VM fetch and direct data counters. Replace CUTLASS's outer
tile scheduler/CLC acquisition with the VM-supplied cluster tile range, but keep
its device mainloop, epilogue, layout and pipeline state machinery. Do not call
`GemmUniversalAdapter::run`, cuBLAS, or nested kernels from native forward.

Shared-memory budget includes VM metadata. If standalone stage depth cannot
fit alongside it, reduce/alias VM metadata or document and measure the changed
collective configuration; do not claim the original pipeline was retained.

Touch points: `include/megakernel.cuh`, `util.cuh`, controller dispatch,
backend configuration/launch validation, and new collective/cluster adapters.
Keep the reference/Llama configuration paths unchanged.

## P2 — The selected prefix GEMM and GeGLU collectives

First implement the default unquantized Thor prefix path, not the union of all
experimental tactics. Tile shapes below are the CUTLASS declared shapes; they
must not be interpreted as the amount computed independently by every CTA.

| Site | Source target | Declared tile M/N/K | Cluster | Required native work |
|---|---|---|---|---|
| Prefix QKV | `sm100_fp16_k64` / `cutlass_fp16_k64` | 256/256/64 | 2/2/1 | BF16 Main/Epi collectives, resolved deep pipeline, QKV output |
| Prefix O | `sm100_fp16_plain` / `cutlass_fp16_plain` | 256/128/64 | 2/2/1 | BF16 collective with beta=1 residual epilogue |
| Prefix gate/up | `FlashRtMegakernelGeGLUG8FusedGemm` | 128/128/128 | 2/2/1 | Two phases with shared A staging, one TMEM allocation, shared-memory gate auxiliary |
| Prefix down | `sm100_fp16_2sm21` / `cutlass_fp16_2sm21` | 256/256/128 | 2/1/1 | Explicit two-SM mainloop/epilogue, beta=1 residual |

For GeGLU, port `CollectiveMmaGate`, `CollectiveEpiGate`, `CollectiveMmaUp`,
`CollectiveEpiUp`, and the visitors from `sm100_smem_aux_visitor.hpp` with BF16
storage. Preserve the explicit three-stage mainloops. Gate applies the actual
`GELU_taylor` functor and stores its output in shared memory; up consumes it in
the multiply epilogue. This is not the current interleaved-row wide GEMM.

The bundled GeGLU+down entry launches two kernels in standalone FlashRT. Native
VM implements its two dependent device operations in the same VM launch; do not
mistake the single C entry for an already fused down-projection kernel.
Similarly `flashrt_rms_qkv_fp16` is a two-operation bundle, not a fused device
RMS+GEMM kernel. Initially preserve these operation boundaries.

Sources: `csrc/gemm/gemm_types_sm100_fp16.h`, `cutlass_sm100_fp16.cu`,
`csrc/gemm/mega/flashrt_megakernel_geglu_g8.cu`,
`flashrt_megakernel_geglu_g8_kernel.hpp`, `sm100_smem_aux_visitor.hpp`.

Only port `sq`, `wide`, `t1`, or separate GELU/mul-aux alternatives if the BF16
shadow actually selects them or a measured ablation motivates adding them.

## P3 — Device replacements for the opaque GEMM sites

These are cuBLAS/cuBLASLt calls in FlashRT. Implement BF16 tensor-core device
operators and select their tactics against the shadow. A source-level exact
copy of their internal kernels is not available.

Let V be views, S=256V+prompt length, A=action horizon and R=A+1:

| Site | M | N | K | Required behavior |
|---|---:|---:|---:|---|
| Patch projection | 256V | 1152 | 588 | Real patch layout, tail handling, tensor-core GEMM |
| Vision QKV | 256V | 3456 | 1152 | Three concatenated projections; separate bias rounding |
| Vision O | 256V | 1152 | 1152 | Projection followed by strict bias/residual behavior |
| Vision FC1 | 256V | 4304 | 1152 | Projection, bias and GELU in source order |
| Vision FC2 | 256V | 1152 | 4304 | Projection followed by strict bias/residual behavior |
| Vision projector | 256V | 2048 | 1152 | Post-LayerNorm GEMM and bias |
| State projection | 1 | 1024 | 32 | Once per request, bias |
| Action input | A | 1024 | 32 | Once per step, bias |
| Action/time action half | A | 1024 | 1024 | Add precomputed time branch and apply SiLU |
| Action/time output | A | 1024 | 1024 | Bias, action-token assembly |
| Expert QKV | R | 2560 | 1024 | BF16 unquantized replacement for source FP8 GEMM |
| Expert O | R | 1024 | 2048 | Projection and residual in decoder source order |
| Expert gate/up | R | 8192 | 1024 | Block-concatenated gate/up, GeGLU |
| Expert down | R | 1024 | 4096 | Projection and residual |
| Action output | A | 32 | 1024 | Source-scaled weight/bias and beta=1 Euler accumulation |

Do not route patch, action/time or final heads through the current scalar FP32
FMA loops. Preserve the true 4304 and 588 extents; padding is an implementation
layout choice with masked/zero-filled tails, not a model shape change.

Sources: `csrc/gemm/gemm_runner.{h,cu}`, `csrc/kernels/decoder_fused.cu`,
`frontends/torch/pi0_thor.py::_patch_embed_ops`,
`models/pi0/pipeline_thor.py::decoder_forward_pi0`.

## P4 — Exact attention paths, including an earlier correction

The default Thor Pi0 backend does NOT use fused FMHA for all attention sites:

| Site | Actual source path | Faithful BF16 native operator sequence |
|---|---|---|
| SigLIP HD72, per-view 256 tokens | `fmha_strided_full` -> `fmha_fp16_strided.cu` | Port the CUTLASS Sm100 FMHA collective, declared tile 256/128/128, NoMask, arbitrary QKV row stride; BF16 operands/output and FP32 QK/PV accumulators |
| Prefix HD256, 8 Q heads / 1 KV head | `attention_qkv_fp16` | QK GEMM -> BF16 logits -> source softmax -> BF16 probabilities -> PV GEMM |
| Expert HD256 with state row | `attention_qkv_fp16_state_masked` | Same QK/PV chain with the fused state/padding mask in softmax |

For Gemma, source treats Q as a matrix of S*8 or R*8 query-head rows sharing
one K/V head. Implement that packed-GQA matrix layout rather than assuming
separate per-head FMHA tasks are equivalent. Preserve the source score scaling
and cast boundary. State row's eight head rows see only prefix+state; action
rows see all keys. Preserve logical NK versus even padded logits stride.

The current FA2 Gemma operator is an optional alternative and must not be the
faithful-path default. Its different softmax/rounding and separate state tile
are not copies of the default FlashRT path. FA4 is also an opt-in alternative,
not the selected Pi0 Thor default.

Sources: `hardware/thor/attn_backend.py::run`,
`csrc/kernels/attention_cublas.cu`, `softmax.cu`,
`csrc/attention/fmha_fp16_strided.cu` and its CUTLASS FMHA headers.
Move standalone FMHA workspace/LSE allocation to model setup and use VM-owned
shared/TMEM storage. Remove its standalone `IndividualTileScheduler` in favor
of static VM tile ranges, retaining the actual load/mainloop/epilogue code.

## P5 — Packing, RoPE, normalization and numerical boundaries

These are required for fidelity, even where a mathematically equivalent
operation already exists in the VM:

| Part | Required adaptation | Difference from current backend |
|---|---|---|
| Patch flattening | FlashRT NHWC im2col flattens pixel-row/pixel-column/channel; permute weights consistently | Current reference-style scalar path flattens channel/row/column |
| Vision weights | Preserve logical matrices across the cuBLAS [K,N] convention and native operand layouts | Existing packing is optimized for our general [N,K] projection family |
| Q/K RoPE packing | Apply `_interleave_qk_core` within every head at setup, then concatenate Q/K/V | Current backend retains OpenPI half-split rotary layout |
| RoPE kernel | Adjacent pairs with interleaved cos/sin; FP32 pair expression then BF16 output; correct prefix/suffix positions and cache strides | Current producer rounds individual products and uses half-split pairs |
| RMS affine | Fold `(1+weight)` into QKV/gate/up in the source's FP32-transform then BF16-cast order; runtime RMS uses ones | Current default explicitly normalizes and applies affine |
| Gate/up packing | `[all gate rows; all up rows]`, or distinct views of those matrices for two-phase GeGLU | Current backend interleaves gate/up row pairs |
| LayerNorm/RMSNorm | Port source reduction and affine precision; map its logical 256-thread reduction to native participants without changing its reduction tree | Current numerics follow ATen/Welford/ATen RMS conventions |
| Bias/residual | Site-specific casts: vision uses GEMM output rounding, then strict bias and residual rounding; selected prefix O/down uses beta=1 epilogues | Current reference-style round-before-residual policy is not correct at every FlashRT site |
| GeGLU | Use the source EVT functors and auxiliary storage type/casts | Rounding both separate projections to BF16 before activation/multiply is not necessarily the source EVT order |
| Action/time preprocessing | Source action_in -> bias -> action-half GEMM -> add time/SiLU -> output GEMM -> bias | Current FP32-composed 32->1024 weight changes arithmetic and rounding |
| Euler update | Source pre-scaled action-output weight/bias and beta=1 accumulation with explicit BF16 output boundaries | Current FP32 head and separate FP32 dt multiply/add preserve OpenPI instead |

Reuse the immutable weight cache, but change its contents/layout to this policy.
Keep timestep biases resident. Do not introduce the existing extra algebraic
folds into the first faithful comparison. Source rules win over a generic
"always round before residual" or "always fuse bias" rule.

Port `patch_im2col`, `patch_embed_bias_pos`, LayerNorm/RMSNorm, bias/residual,
SiLU/GeGLU, QKV/RoPE/cache, language assembly, state/action assembly and output
updates as device functions/range operators. Remove `__global__` launch and
host allocation assumptions. If a source helper uses all 256 threads, emulate
its logical reduction tree in consumers or add a quiescent collective opcode;
do not silently halve the reduction participants.

Sources: `executors/torch_weights.py::{FusedQKV,FusedGateUp}`,
`_thor_spec_common.py`, `csrc/kernels/{patch_embed,norm,rope,activation,
elementwise,softmax,decoder_fused}.cu`, and Pi0 timestep/weight setup.

## P6 — Native model compiler and memory program

Replace the current generic linear-op selection with an explicit per-site
tactic/epilogue manifest. Instructions reference prepared collective operands
and a logical tile range; cluster members share operation identity.

Recompute task coverage and producer expectations from each collective's true
output partition. A multi-CTA tile must not be counted as independent complete
tiles by every member. Publish after all its participating stores complete.
Keep prefix KV once per request and suffix KV in its own region per layer.
Maintain ready/acquire ordering for all ten steps and changed-input replay.

Port the source's scratch-buffer lifetimes and aliases: in-place residual `x`,
normalization scratch, QKV, attention/logit scratch, gate/hidden/FFN output,
action/time scratch and the action/noise accumulator. Reuse them across layers
and steps as FlashRT does instead of retaining a separate activation allocation
for every stage. Keep the prefix KV cache resident and distinct from scratch.
This working-set/data-reuse requirement is mandatory for a faithful performance
comparison, not an optional memory optimization.

Reuse requires explicit last-reader dependencies, not just producer readiness.
A next layer/step must not overwrite storage still consumed by another worker
or cluster. Initially the source's sequential operation boundaries provide a
simple drain point; finer overlap can be added later. Disable per-layer retained
activation snapshots in performance mode. Debug snapshots require a separate
diagnostic mode with its overhead excluded from performance claims. Setup-once
immutable packing is already reusable.

Preserve last-prefix dead-output pruning only where equivalence to the shadow
is established. Keep the existing reference's additional fusions as optional
experiments until compared separately. Their presence is not evidence of a
faithful FlashRT lowering.

Touch points: `flashrt_pi0/model.py`, `program` generation, `model.cu`,
`native.cu`, `dense.cu`, descriptor setup and launch/context ownership.

## Implementation order and acceptance

1. Build P0 shadow/manifest and one minimal P1 cluster/collective integration.
2. Port prefix QKV k64, O plain, two-phase GeGLU and down 2sm21. These address
   the measured approximately 3x prefix projection gap. Measure the same BF16
   collective standalone and inside the VM before broad model integration.
3. Replace patch/vision/projector and state/action/time/output scalar or general
   replacements with the P3 tensor-core implementations; use measured tactics.
4. Port P4 attention and P5 support kernels/packing, then emit the complete P6
   model program with all ten steps in one native launch.
5. Compare full forward to the BF16 shadow, then separately to OpenPI. Record
   any policy-induced OpenPI error rather than changing numerical boundaries
   covertly. Run the existing default/prompt/small replay checks and focused
   memory/synchronization checks for the new cluster/resource paths.
6. Compare equal workloads and timing boundaries: standalone BF16 shadow versus
   native, same fixed inputs, isolated GPU, raw samples and launch trace. Report
   collective time, scheduler/dependency waits and end-to-end latency. Do not
   declare kernel equivalence while an unexplained multi-fold gap remains.

Existing validated components to keep: complete ordinary-pi0 wiring as a
structural reference; static instruction serialization; direct producer counters;
replay validation; one-launch tracing; same-weight OpenPI harness; immutable
weight ownership; pinned dependency/build fingerprints.

Completion requires both a documented BF16 FlashRT compute contract and its
native execution. The present 262 ms backend satisfies neither full collective
coverage nor a matched FlashRT BF16 performance comparison.
